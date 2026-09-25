"""Decode videos with OpenCV and create centered frame context windows."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
from detection.utils import json_save



def natural_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)',path.name)]


def image_files(folder, cfg):
    suffixes = set(cfg['image_extensions'])
    paths = sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in suffixes), key=natural_key)
    if not paths:
        raise ValueError(f'No frames in {folder}')
    return paths


def spatial_transform(rgb, cfg):
    import cv2
    size = cfg['size']
    methods = dict(cubic=cv2.INTER_CUBIC,linear=cv2.INTER_LINEAR,area=cv2.INTER_AREA)
    interpolation = methods[cfg['interpolation']]
    if cfg['spatial_transform']=='square':
        return cv2.resize(rgb,(size,size),interpolation=interpolation)
    h,w = rgb.shape[:2]
    scale = max(size,cfg['crop_resize_short_side'])/min(h,w)
    rgb = cv2.resize(rgb,(round(w*scale),round(h*scale)),interpolation=interpolation)
    h,w = rgb.shape[:2]
    top,left = (h-size)//2,(w-size)//2
    return rgb[top:top+size,left:left+size]


def frame_timestamps(folder, paths, cfg):
    import pandas as pd
    folder = Path(folder)
    metadata = folder/'metadata.json'
    info = json.loads(metadata.read_text()) if metadata.exists() else {}
    if 'frames' in info and info['frames'] != len(paths):
        raise ValueError('Frame count does not match completion metadata')
    timestamp_path = folder/'timestamps.csv'
    if timestamp_path.exists():
        table = pd.read_csv(timestamp_path)
        lookup = dict(zip(table.filename, table.timestamp_seconds))
        if all(p.name in lookup for p in paths):
            times = np.array([lookup[p.name] for p in paths],dtype=float)
            if np.isnan(times).all():
                return times
            if not np.isfinite(times).all() or np.any(np.diff(times)<0):
                raise ValueError('Invalid frame timestamps')
            return times
        raise ValueError('timestamps.csv does not cover every selected frame')
    fps = cfg['fallback_fps']
    if metadata.exists():
        fps = info.get('fps',fps)
    return np.arange(len(paths))/float(fps) if fps and fps>0 else np.full(len(paths),np.nan)


def preprocess(source, output, cfg):
    """Materialize PNGs using the current cubic square resize by default."""
    import cv2
    import pandas as pd
    from PIL import Image
    source,output = Path(source),Path(output)
    if source.resolve()==output.resolve():
        raise ValueError('Preprocessing output must differ from input')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preprocessing requires an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    names, times = [],[]
    if source.is_dir() or source.suffix.lower() in cfg['image_extensions']:
        paths = image_files(source,cfg) if source.is_dir() else [source]
        stamps = frame_timestamps(source,paths,cfg) if source.is_dir() else [0.0]
        for i,(path,stamp) in enumerate(zip(paths,stamps),1):
            with Image.open(path) as image:
                rgb = spatial_transform(np.asarray(image.convert('RGB')),cfg)
            filename = f'frame_{i:06d}.png'
            if not cv2.imwrite(str(output/filename),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
                raise OSError('Failed to save frame')
            names.append(filename)
            times.append(float(stamp))
    else:
        capture = cv2.VideoCapture(str(source))
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened():
            raise ValueError(f'Cannot open {source}')
        try:
            while True:
                ok,bgr = capture.read()
                if not ok:
                    break
                rgb = spatial_transform(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),cfg)
                filename = f'frame_{len(names)+1:06d}.png'
                if not cv2.imwrite(str(output/filename),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
                    raise OSError('Failed to save frame')
                names.append(filename)
                times.append(capture.get(cv2.CAP_PROP_POS_MSEC)/1000)
        finally:
            capture.release()
        if expected>0 and len(names)!=expected:
            raise ValueError('Incomplete video decode')
    if not names:
        raise ValueError('No decoded frames')
    # Missing timestamps remain blank; frame-based scoring remains possible.
    pd.DataFrame(dict(frame_index=np.arange(len(names)),filename=names,timestamp_seconds=times)).to_csv(output/'timestamps.csv',index=False)
    json_save(output/'metadata.json',dict(frames=len(names),source=str(source.resolve()),preprocessing=cfg))


def create_context_windows(frame_count, window_size=5, *, stride=1):
    """Return zero-based frame-index windows and centers with repeated edges.

    Odd-width windows are centered on each selected frame. Repeating the first
    and last frames preserves boundary frames, unlike unpadded series windows.
    """
    for name, value in (('frame_count', frame_count), ('window_size', window_size), ('stride', stride)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if window_size % 2 == 0:
        raise ValueError('Video context window size must be odd')
    centers = np.arange(0, frame_count, stride, dtype=np.int64)
    offsets = np.arange(window_size) - window_size // 2
    return np.clip(centers[:, None] + offsets, 0, frame_count - 1), centers


def mean_embeddings(values, width=5):
    """Average frame embeddings over centered video context windows."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError('Expected finite (time, features) embeddings')
    windows, _ = create_context_windows(len(values), width)
    pooled = np.zeros_like(values)
    for position in range(width):
        pooled += values[windows[:, position]]
    return pooled / width


def main():
    """Write decoded frames, timestamps, and context-window frame indices."""
    from detection.defaults import vision_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--context-length', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    create_context_windows(1, args.context_length, stride=args.stride)
    cfg = vision_config({'model': {'checkpoint': ''}})['preprocessing']
    preprocess(args.input, args.output, cfg)
    paths = image_files(args.output, cfg)
    windows, centers = create_context_windows(len(paths), args.context_length, stride=args.stride)
    np.savez_compressed(Path(args.output) / 'context_windows.npz',
        frame_indices=windows, source_indices=centers,
        filenames=np.asarray([p.name for p in paths]))


if __name__ == '__main__':
    main()
