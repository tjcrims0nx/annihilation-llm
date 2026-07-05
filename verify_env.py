import sys
import subprocess

def main():
    is_gpu = '--gpu' in sys.argv
    needs_reinstall = False
    needs_install = False

    try:
        import torch
        if is_gpu and not torch.cuda.is_available():
            print("CRITICAL: GPU detected but PyTorch is CPU-only! Forcing CUDA reinstall...", flush=True)
            needs_reinstall = True
        
        # Test if torchvision is available
        import torchvision
    except ImportError as e:
        print(f"Missing dependency detected ({e}). Installing...", flush=True)
        needs_install = True

    if needs_reinstall:
        cmd = ['uv', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu121', '--reinstall']
        print(f"Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        print("CUDA PyTorch reinstallation complete.", flush=True)
    elif needs_install:
        cmd = ['uv', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']
        if is_gpu:
            cmd.extend(['--index-url', 'https://download.pytorch.org/whl/cu121'])
        print(f"Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        print("Dependencies installation complete.", flush=True)
    else:
        print("Environment verification passed! All dependencies correctly installed.", flush=True)

if __name__ == '__main__':
    main()
