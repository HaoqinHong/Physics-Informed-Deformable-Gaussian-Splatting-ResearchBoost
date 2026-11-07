"""CPU formulation-only public-data fingerprint; run before freezing the task."""
import argparse
from common import config, data_files, digest, sources, write_json
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    c = config(a.config)
    scenes = {name: data_files(c, name) for name in c['scenes']}
    write_json(c['data_manifest'], {'scenes': scenes, 'sha256': digest(scenes), 'reference_sources': sources(c['reference'])})
if __name__ == '__main__':
    main()
