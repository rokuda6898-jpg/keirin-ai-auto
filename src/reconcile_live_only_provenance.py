"""Explicit migration for reviewed live-only changes, never a loader bypass.

These two functions are not used by fixed-year/three-year fitting or scoring.
Every other AST node must be identical to the original recorded source. The
original hashes and comparison proof remain in the manifest; loaders stay strict.
"""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_ONLY = {'annual_knowledge.py': 'forecast_departments',
             'department_ticket_v2.py': 'public_preserved'}
TARGETS = {
    'models/fusion_shadow_refit_first_anchor.json': [('model_sha256','models/fusion_shadow_refit_first_anchor.joblib')],
    'models/individual_three_year/manifest.json': [('model_sha256','models/individual_three_year/model.joblib')],
    'models/individual_stage/stage_manifest.json': [('source_model_sha256','models/individual_stage/frozen_model.joblib'),
                                                 ('stage_model_sha256','models/individual_stage/stage_model.json')],
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unchanged_training_ast(before, after, ignored_function):
    def retained(source):
        tree = ast.parse(source)
        removed = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == ignored_function]
        if len(removed) != 1:
            raise ValueError('Expected one live-only function')
        tree.body = [n for n in tree.body if n not in removed]
        return ast.dump(tree, include_attributes=False)
    old, new = retained(before), retained(after)
    if old != new:
        raise ValueError('Training or shared code changed; retraining required')
    return digest(new.encode())


def migrate(reference, root=ROOT):
    updates = []
    for name, models in TARGETS.items():
        path = root/name
        manifest = json.loads(path.read_text(encoding='utf-8'))
        for key, model in models:
            data = (root/model).read_bytes()
            if model.endswith('.json'):
                data = data.replace(b'\r\n',b'\n')
            if manifest[key] != digest(data):
                raise ValueError('Model binary changed: '+model)
        hashes = dict(manifest['source_hashes'])
        proof = {}
        for source, expected in hashes.items():
            current = (root/'src'/source).read_bytes().replace(b'\r\n',b'\n')
            if digest(current) == expected:
                continue
            if source not in LIVE_ONLY:
                raise ValueError('Unreviewed source change: '+source)
            original = subprocess.check_output(['git','show',reference+':src/'+source],cwd=root).replace(b'\r\n',b'\n')
            if digest(original) != expected:
                raise ValueError('Reference does not match recorded model provenance: '+source)
            ast_hash = unchanged_training_ast(original,current,LIVE_ONLY[source])
            proof[source] = {'old_source_sha256':expected,'new_source_sha256':digest(current),
                             'unchanged_training_ast_sha256':ast_hash,'changed_live_function':LIVE_ONLY[source]}
            hashes[source] = digest(current)
        if proof:
            manifest['source_hashes'] = hashes
            manifest.setdefault('live_only_compatibility',[]).append({'reference_commit':reference,'proof':proof,
                'retrained':False,'model_bytes_unchanged':True})
            updates.append((path,manifest))
    # Validate all artifacts before modifying any manifest.
    for path,manifest in updates:
        path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return [str(p.relative_to(root)) for p,_ in updates]


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',required=True)
    print(json.dumps(migrate(parser.parse_args().reference)))
