import argparse
from cleanfid import fid

def main(args):
    score = fid.compute_fid(args.dir1, args.dir2)
    print(f'FID score: {score}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FID score between two directories.')
    parser.add_argument('--dir1', type=str, required=True, help='Path to the first directory')
    parser.add_argument('--dir2', type=str, required=True, help='Path to the second directory')
    args = parser.parse_args()
    main(args)
