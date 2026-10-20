"""Exact, reviewed storage-only compatibility for an existing frozen model.

The training manifest remains immutable. This is not a general hash override:
only the recorded legacy manifests and their reviewed source replacements are
accepted. All other dependencies, the model bytes and the date guard retain
their original checks. Source changes outside storage and the loader hook are
also compared with the legacy module AST at runtime.
"""
import ast
import hashlib
import json
from pathlib import Path

PROOF_PATH = Path('models/individual_three_year/storage_compatibility.json')
LEGACY_MANIFEST_SHA256 = '21595604842c84460f5041d5c6849635aaeaab65b8ef7a1f3921c95b5e29b530'
LEGACY_SOURCES = {
    'department_experiment_v2.py': '15fb9067140319a5875e52216afb1f805409db2041da69e2e056a78281b857d9',
    'individual_three_year.py': '73214c8a04553b7c3e67445578626952851627c3cf6838e1d1d70d9599913c09',
}
SHADOW_PROOF_PATH = Path('models/fusion_shadow_storage_compatibility.json')
SHADOW_LEGACY_SOURCES = {
    'department_experiment_v2.py': LEGACY_SOURCES['department_experiment_v2.py'],
    'fusion_shadow_live.py': 'a57113bd793222eda71effdb892085df53db68ecfe4cd420b71198b70fe557b8',
}
PROFILES = {
    'three_year': {
        'proof_path': PROOF_PATH,
        'manifest_path': Path('models/individual_three_year/manifest.json'),
        'manifest_sha256': LEGACY_MANIFEST_SHA256,
        'schema': 'three-year-storage-compatibility-v1',
        'sources': LEGACY_SOURCES,
    },
    'fusion_shadow': {
        'proof_path': SHADOW_PROOF_PATH,
        'manifest_path': Path('models/fusion_shadow_refit_first_anchor.json'),
        'manifest_sha256': '9ef36fcfb345a3e58b44a6794417ed36242132b2ba11004208d81db8207ffd17',
        'schema': 'fusion-shadow-storage-compatibility-v1',
        'sources': SHADOW_LEGACY_SOURCES,
        'pipeline_sha256': '70fb57cb29d6a4c8624a27177eafac17ff8c5f5081f17ea9b41fae465e5f2e46',
    },
}
ALLOWED_EDITS = {
    'department_experiment_v2.py': {
        'functions': ('append', 'read_lines'),
        'added_import': 'from ledger_io import append_ledger_line, is_archived_ledger, ledger_lines',
    },
    'individual_three_year.py': {
        'functions': ('load',),
        'added_import': 'import model_storage_compat as storage_compat',
    },
    'fusion_shadow_live.py': {
        'functions': ('load_model',),
        'added_import': 'import model_storage_compat as storage_compat',
    },
}


def source_bytes(path):
    return Path(path).read_bytes().replace(b'\r\n', b'\n')


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def _ast_value(value):
    if isinstance(value, ast.AST):
        # Python 3.12 adds an empty type_params field to ordinary functions;
        # omitting only this empty field keeps the certificate portable.
        return [type(value).__name__, [[name, _ast_value(item)]
                for name, item in ast.iter_fields(value)
                if not (name == 'type_params' and item == [])]]
    if isinstance(value, list):
        return [_ast_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return [type(value).__name__, repr(value)]


def stable_ast_sha256(name, source):
    """Hash the entire module except the explicitly reviewed storage hook."""
    rule = ALLOWED_EDITS[name]
    tree = ast.parse(source)
    added_import = _ast_value(ast.parse(rule['added_import']).body[0])
    removed = []
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in rule['functions']:
            removed.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and _ast_value(node) == added_import:
            continue
        else:
            body.append(node)
    if sorted(removed) != sorted(rule['functions']):
        raise ValueError('storage compatibility hook definitions changed')
    tree.body = body
    canonical = json.dumps(_ast_value(tree), ensure_ascii=True, separators=(',', ':'))
    return sha256(canonical.encode('utf-8'))


def _certificate_match(manifest, current_hashes, *, root, manifest_path,
                       profile_name, pipeline_sha256=None):
    """Validate only the immutable manifest and source pairs for this profile."""
    recorded = manifest.get('source_hashes')
    if not isinstance(recorded, dict) or not recorded or set(recorded) != set(current_hashes):
        return False
    try:
        root = Path(root)
        profile = PROFILES[profile_name]
        proof = json.loads((root / profile['proof_path']).read_text(encoding='utf-8'))
        raw_manifest = Path(manifest_path).read_bytes()
        if (proof['schema'] != profile['schema']
                or proof['original_manifest_sha256'] != profile['manifest_sha256']
                or sha256(raw_manifest) != profile['manifest_sha256']
                or json.loads(raw_manifest) != manifest
                or set(proof['sources']) != set(profile['sources'])):
            return False
        for field in ('version', 'model_sha256', 'training_cutoff_exclusive'):
            if proof[field] != manifest[field]:
                return False
        verifier_sha = sha256(source_bytes(__file__))
        if (proof['verifier_sha256'] != verifier_sha
                or sha256(source_bytes(root / 'src/model_storage_compat.py')) != verifier_sha):
            return False
        for name, old_sha in recorded.items():
            if name not in profile['sources'] and current_hashes[name] != old_sha:
                return False
        for name, old_sha in profile['sources'].items():
            change = proof['sources'][name]
            current_source = source_bytes(root / 'src' / name)
            if (change['before_sha256'] != old_sha
                    or change['after_sha256'] == old_sha
                    or sha256(current_source) != change['after_sha256']
                    or stable_ast_sha256(name, current_source) != change['unchanged_ast_sha256']):
                return False
            if name in recorded and (recorded[name] != old_sha
                                      or current_hashes[name] != change['after_sha256']):
                return False
        if 'pipeline_sha256' in profile:
            pipeline = proof['pipeline']
            actual_pipeline = shadow_pipeline_sha256(source_bytes(root / 'src/fusion_shadow_live.py'))
            if (manifest['model_pipeline_hash'] != profile['pipeline_sha256']
                    or pipeline['before_sha256'] != profile['pipeline_sha256']
                    or pipeline['after_sha256'] != pipeline_sha256
                    or actual_pipeline != pipeline_sha256):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, SyntaxError):
        return False


def source_hashes_match(manifest, current_hashes, *, root, manifest_path):
    """Use ordinary exact three-year provenance, or its reviewed transition."""
    recorded = manifest.get('source_hashes')
    if isinstance(recorded, dict) and recorded and recorded == current_hashes:
        return True
    return _certificate_match(manifest, current_hashes, root=root,
        manifest_path=manifest_path, profile_name='three_year')


def shadow_pipeline_sha256(source):
    """Exactly the existing shadow fingerprint, including the guarded loader."""
    relevant = {'dated_phases', 'digest', 'load_model', 'rank_race_candidates',
                'source_hashes', 'train'}
    payload = '\n'.join(ast.dump(node, include_attributes=False) for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in relevant)
    return sha256(payload.encode('utf-8'))


def fusion_shadow_match(manifest, current_hashes, current_pipeline, *, root, manifest_path):
    """Allow only the reviewed shadow dependency and loader-pipeline pair."""
    return _certificate_match(manifest, current_hashes, root=root,
        manifest_path=manifest_path, profile_name='fusion_shadow', pipeline_sha256=current_pipeline)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    checks = {}
    for name, profile in PROFILES.items():
        manifest_path = root / profile['manifest_path']
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        current = {item: sha256(source_bytes(root / 'src' / item)) for item in manifest['source_hashes']}
        pipeline = shadow_pipeline_sha256(source_bytes(root / 'src/fusion_shadow_live.py')) if name == 'fusion_shadow' else None
        if not _certificate_match(manifest, current, root=root, manifest_path=manifest_path,
                                  profile_name=name, pipeline_sha256=pipeline):
            raise SystemExit(f'{name} storage compatibility verification failed')
        checks[name] = {'source_and_AST_compatibility_verified': True,
                       'original_manifest_sha256': sha256(manifest_path.read_bytes()),
                       'source_changes': sorted(item for item in current
                                                if current[item] != manifest['source_hashes'][item])}
    print(json.dumps(checks))
