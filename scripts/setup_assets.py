"""Fetch and verify the pinned upstream sources and inference assets."""

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
HUB = 'assets/torch/hub'
MODEL = 'assets/talk2dino-decap'
SNAPSHOT = ('https://huggingface.co/Ruggero1912/Patch-ioner_talk2dino_decap_COCO_Captions'
            '/resolve/b2ff74dc8f81b5182576dc72db408e113d6c6bce')
REPOSITORIES = (
    ('https://github.com/Ruggero1912/Patch-ioner.git', 'upstream', '34d8130dafba0e4bb148bd0912c4a9186e398a05'),
    ('https://github.com/facebookresearch/dinov2.git', f'{HUB}/facebookresearch_dinov2_main',
     '7764ea0f912e53c92e82eb78a2a1631e92725fc8'),
)
DECODER = 'coco_karpathy-009.pt'
MEMORY = 'coco_captions_text_embeddings-B16-ViT-B.16-591753.h5'
DINO = 'dinov2_vitb14_reg4_pretrain.pth'
EXAMPLE = ('https://huggingface.co/spaces/Ruggero1912/Patch-ioner'
           '/resolve/7bc37c3f9da0a8b2742daf2d924faea818d75b0e/example-images/dog-frisbee.jpg')
FILES = (
    (f'{SNAPSHOT}/config.yaml', f'{MODEL}/config.yaml', 766,
     '2a24aee34b109f2f72db00f4cb2b701db18ab5f2505fa6eba9458c4a917412f7'),
    (f'{SNAPSHOT}/{DECODER}', f'{MODEL}/{DECODER}', 273331640,
     '2e0cbdb225201f325bfdf3f821b6cfca396180b49ae95334aa1461979165a763'),
    (f'{SNAPSHOT}/{MEMORY}', f'{MODEL}/{MEMORY}', 1869966480,
     '68ac0ffde71488999f13564cd417a12b69435a437f8a76cbf9a89966f7bb2e83'),
    (f'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/{DINO}', f'{HUB}/checkpoints/{DINO}', 346393545,
     '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71'),
    (EXAMPLE, 'assets/examples/dog-frisbee.jpg', 114268,
     '6c88c5b68110d16250cd20a01d6dd46cd4d815a8a43f7fe6cd1156d55c23828e'),
)


def verified(path, size, checksum):
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == checksum


def checkout(url, path, commit, verify_only, proxy):
    if not path.exists() and not verify_only:
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f'.{path.name}-', dir=path.parent) as temporary:
            clone = Path(temporary) / 'repo'
            git = ['git', '-c', f'http.proxy={proxy}'] if proxy else ['git']
            subprocess.run([*git, 'clone', '--no-checkout', url, str(clone)], check=True)
            subprocess.run(['git', '-C', str(clone), 'checkout', '--detach', commit], check=True)
            clone.rename(path)
    if not (path / '.git').exists():
        raise RuntimeError(f'Missing checkout at {path}')
    current = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if current != commit:
        raise RuntimeError(f'{path} is at {current}. Expected {commit}. Existing checkout was kept.')
    print(f'Verified checkout {path}', flush=True)


def download(url, path, size, checksum, verify_only, proxy):
    if verified(path, size, checksum):
        print(f'Verified {path}', flush=True)
        return
    if path.exists() or verify_only:
        raise RuntimeError(f'Missing or invalid asset: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f'{path.name}.part')
    if not verified(partial, size, checksum):
        if partial.exists() and partial.stat().st_size >= size:
            raise RuntimeError(f'Invalid partial download: {partial}')
        options = ['--proxy', proxy] if proxy else []
        subprocess.run(['curl', '--fail', '--location', '--retry', '2', '--connect-timeout', '15',
                        '--continue-at', '-', '--output', str(partial), *options, url], check=True)
        if not verified(partial, size, checksum):
            raise RuntimeError(f'Download verification failed: {partial}')
    partial.rename(path)
    print(f'Verified {path}', flush=True)


def install_dependencies(proxy):
    for package, expected in (('torch', '2.5.1'), ('torchvision', '0.20.1')):
        try:
            actual = version(package).split('+', 1)[0]
        except PackageNotFoundError:
            actual = None
        if actual != expected:
            raise RuntimeError(f'Set up the CUDA build of {package}=={expected} in the active environment first.')
    import torch

    if torch.version.cuda is None:
        raise RuntimeError('The active environment has a CPU-only PyTorch. Install a CUDA build first.')
    options = ['--proxy', proxy] if proxy else []
    subprocess.run([sys.executable, '-m', 'pip', 'install', *options, '-r', str(ROOT / 'requirements.txt')], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    if not shutil.which('java'):
        print('For caption evaluation, add Java to PATH for PTB tokenization and METEOR.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--assets-only', action='store_true', help='Fetch sources and assets without installing packages')
    mode.add_argument('--verify-assets', action='store_true', help='Verify local sources and assets offline')
    mode.add_argument('--dependencies-only', action='store_true', help='Install packages in the active CUDA environment')
    parser.add_argument('--proxy', help='Override the HTTP proxy used for Git, curl and pip')
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error('Python 3.10 or newer is required')
    try:
        if not (args.assets_only or args.verify_assets):
            install_dependencies(args.proxy)
        if not args.dependencies_only:
            for url, path, commit in REPOSITORIES:
                checkout(url, ROOT / path, commit, args.verify_assets, args.proxy)
            for url, path, size, checksum in FILES:
                download(url, ROOT / path, size, checksum, args.verify_assets, args.proxy)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
