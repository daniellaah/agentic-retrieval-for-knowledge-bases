"""Read YAML frontmatter properties without changing snapshot text or offsets."""

from dataclasses import dataclass
import yaml

from .sources import KnowledgeSnapshot, SourceRef


@dataclass(frozen=True)
class NoteMetadata:
    source: SourceRef
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


def read_metadata(snapshot: KnowledgeSnapshot, source: SourceRef) -> NoteMetadata:
    text = snapshot.read_note(source).content
    lines = text.splitlines()
    properties = {}
    if lines and lines[0].strip() == '---':
        end = next((i for i in range(1, len(lines)) if lines[i].strip() in ('---', '...')), None)
        if end is None:
            raise ValueError(f'Unclosed frontmatter in {source.path}.')
        try:
            properties = yaml.load('\n'.join(lines[1:end]), Loader=_UniqueSafeLoader)
            if properties is None:
                properties = {}
        except (yaml.YAMLError, ValueError, TypeError) as error:
            raise ValueError(f'Invalid frontmatter in {source.path}.') from error
        if not isinstance(properties, dict):
            raise ValueError(f'Frontmatter must be a mapping in {source.path}.')
    return NoteMetadata(source, _strings(properties.get('tags'), 'tags'),
                        _strings(properties.get('aliases'), 'aliases'))


def _strings(value, name):
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(f'{name} must be a string or a list of nonblank strings.')
    return tuple(dict.fromkeys(v.strip() for v in value))


class _UniqueSafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(set(keys)) != len(keys):
            raise ValueError('Frontmatter contains duplicate keys.')
        return super().construct_mapping(node, deep=deep)
