import json
from pathlib import Path
import numpy as np
from video_preprocessing import image_files, frame_timestamps, spatial_transform, mean_embeddings
from .utils import json_save
SUPPORTED_MODELS = ('dino_vits16','dino_vitb16','dinov2_vits14','dinov2_vitb14')


def load_model(name, cfg, cache):
    import torch
    if name not in SUPPORTED_MODELS:
        raise ValueError(f'Unknown model: {name}')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.manual_seed(cfg['seed'])
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg['device']=='auto' else cfg['device']
    repository = cfg['dinov2_repository'] if name.startswith('dinov2') else cfg['dino_repository']
    if not Path(repository).is_dir():
        raise FileNotFoundError('A local DINO repository checkout is required')
    model = torch.hub.load(repository,name,source='local',pretrained=False)
    model.load_state_dict(torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True))
    return model.eval().to(device),device


def embed(folder, output, name, config, loaded=None):
    import torch
    from PIL import Image
    cfg = config['preprocessing']
    paths = image_files(folder,cfg)
    times = frame_timestamps(folder,paths,cfg)
    meta_path=Path(folder)/'metadata.json'
    materialized=meta_path.exists() and json.loads(meta_path.read_text()).get('preprocessing')==cfg
    model,device = loaded or load_model(name,config['embedding'],config['paths']['model_cache'])
    batch_size = config['embedding']['batch_size']
    mean = torch.tensor(cfg['normalization_mean']).view(3,1,1)
    std = torch.tensor(cfg['normalization_std']).view(3,1,1)
    outputs = []
    with torch.inference_mode():
        for start in range(0,len(paths),batch_size):
            tensors=[]
            for path in paths[start:start+batch_size]:
                with Image.open(path) as image:
                    rgb=np.asarray(image.convert('RGB'))
                    if not materialized:
                        rgb=spatial_transform(rgb,cfg)
                    elif rgb.shape[:2]!=(cfg['size'],cfg['size']):
                        raise ValueError('Preprocessed frame has unexpected size')
                tensor=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
                tensors.append((tensor-mean)/std)
            values=model(torch.stack(tensors).to(device))
            if values.ndim!=2 or len(values)!=len(tensors):
                raise ValueError('Model did not return one embedding per image')
            outputs.append(values.cpu().numpy())
    single=np.concatenate(outputs)
    if not np.isfinite(single).all():
        raise ValueError('Nonfinite model embeddings')
    aggregate=mean_embeddings(single,config['embedding']['aggregation']['width'])
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix('.npz.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle,single_frame_embeddings=single,embeddings=aggregate,
                            frame_numbers=np.arange(1,len(single)+1),timestamps=times)
    temp.replace(output)
    json_save(output.with_suffix('.json'),dict(model=name,frames=len(single),dimension=single.shape[1],
        source=str(Path(folder).resolve()),preprocessing=cfg,embedding=config['embedding'],device=device))
