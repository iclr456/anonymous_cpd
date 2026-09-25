"""Vision adapter settings, with local checkpoints only."""
def vision_config(job):
    model = job['model']
    embedding = dict(device='cpu', batch_size=5, cpu_threads=4, seed=0,
        precision='float32', dino_repository=model.get('repository', ''),
        dinov2_repository=model.get('repository', ''), checkpoint=model['checkpoint'],
        aggregation={'method': 'mean', 'width': 5, 'stride': 1, 'alignment': 'centered', 'padding': 'nearest'})
    embedding.update(job.get('embedding', {}))
    aggregation = embedding['aggregation']
    if any(aggregation[k] != v for k, v in dict(method='mean', stride=1, alignment='centered', padding='nearest').items()):
        raise ValueError('Vision aggregation requires centered mean, stride 1, nearest padding')
    if embedding['batch_size'] < 1 or embedding['cpu_threads'] < 1 or embedding['precision'] != 'float32':
        raise ValueError('Invalid vision batch size, CPU threads or precision')
    preprocessing = dict(size=224, spatial_transform='square', crop_resize_short_side=256,
        interpolation='cubic', materialize_frames=False,
        image_extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'],
        fallback_fps=None, normalization_mean=[.485, .456, .406], normalization_std=[.229, .224, .225])
    preprocessing.update(job.get('preprocessing', {}))
    if preprocessing['size'] != 224 or len(preprocessing['normalization_mean']) != 3 or len(preprocessing['normalization_std']) != 3 or min(preprocessing['normalization_std']) <= 0:
        raise ValueError('Vision requires 224x224 RGB inputs and valid normalization')
    return dict(embedding=embedding, preprocessing=preprocessing, paths={'model_cache': '.'})
