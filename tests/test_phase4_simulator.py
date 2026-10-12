import json
from pathlib import Path

import pytest

from jspace_research.phase4.common import content_hash
from jspace_research.phase4.simulator import atomic_json, resolve_response


def test_native_request_and_validated_resume(tmp_path):
    root = tmp_path / 'benchmark'
    prompts = root / 'src/prompts'
    prompts.mkdir(parents=True)
    (prompts / 'generation_prompts.py').write_text(
        "SYSTEM_MESSAGE='system'\nEXAMPLE='example'\n"
        "DS_ATTACKER_TOOL_RESPONSE_GEN_MESSAGE='{example}|{attacker_tool}|{tool_parameters}'\n"
    )
    out = tmp_path / 'run'
    tools = {'Search': {'name': 'Search'}}
    args = (root, out, {}, tools, 'Search', '{"keywords": ["a"]}')
    with pytest.raises(RuntimeError, match='Native simulator request saved'):
        resolve_response(*args)
    pending = json.loads((out / 'injecagent_simulator_pending.json').read_text())
    assert pending['request']['model'] == 'gpt-4-0613'
    assert pending['request']['messages'][1]['content'] == 'example|{\n  "name": "Search"\n}|{"keywords": ["a"]}'
    entry = {'response': 'native result', 'source': 'approved cache',
             'request_sha256': pending['request_sha256'],
             'response_sha256': content_hash('native result')}
    path = out / 'injecagent_simulator_responses.json'
    atomic_json(path, {pending['key']: entry})
    assert resolve_response(*args) == ('native result', 'supplemental')
    entry['response'] = 'tampered'
    atomic_json(path, {pending['key']: entry})
    with pytest.raises(RuntimeError, match='Invalid supplemental'):
        resolve_response(*args)
    entry['response'] = 'native result'
    entry['request_sha256'] = 'wrong'
    atomic_json(path, {pending['key']: entry})
    with pytest.raises(RuntimeError, match='Invalid supplemental'):
        resolve_response(*args)


def test_pinned_response_needs_no_supplement_or_prompt_import(tmp_path):
    assert resolve_response(tmp_path, tmp_path, {'(Search,{})': 'original'}, {}, 'Search', '{}') == ('original', 'pinned')
    assert not (tmp_path / 'injecagent_simulator_pending.json').exists()
