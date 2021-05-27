import argparse
import os
import json
import re
import unicodedata
import bz2
from multiprocessing import Pool
from itertools import chain
from difflib import SequenceMatcher

from edits import EditTagger


invalid_bytes_re = re.compile(r'[\x00-\x1F]+')
sline_re = re.compile(r'\[sline\].*?\[/sline\]')
color_tags = ['[f-blue]','[/f-blue]',
              '[f-red]','[/f-red]',
              '[f-bold]','[/f-bold]']
ja_re = re.compile(r'([ぁ-んァ-ン])')
html_re = re.compile(r'<(\/?[a-z]+)>')
edit_tagger = EditTagger()


def clean_line(line):
    line = unicodedata.normalize('NFKC', line.strip()).replace(' ', '')
    if line.endswith('GOOD'):
        line = line[:-4]
    elif line.endswith('OK'):
        line = line[:-2]
    for tag in color_tags:
        line = line.replace(tag, '')
    line = sline_re.sub('', line).replace('[/sline]', '')
    return line


def preprocess_lang8_part(args,
                          correct_file='corr_sentences.txt',
                          incorrect_file='incorr_sentences.txt',
                          edit_tags_file='edit_tagged_sentences.txt.bz2'):
    rows, part_output_dir = args
    pairs = set()
    for row in rows:
        for learner_sent, corrections in zip(row[4], row[5]):
            if not ja_re.search(learner_sent) or html_re.search(learner_sent):
                continue
            learner_sent = clean_line(learner_sent)
            if not corrections:
                pairs.add((learner_sent, learner_sent))
            else:
                for target_sent in corrections:
                    if not target_sent or not ja_re.search(target_sent) or \
                            html_re.search(target_sent):
                        continue
                    target_sent = clean_line(target_sent)
                    pairs.add((learner_sent, target_sent))
    corr_lines = []
    incorr_lines = []
    edit_lines = []
    for learner_sent, target_sent in pairs:
        # remove appended comments
        matcher = SequenceMatcher(None, learner_sent, target_sent)
        diffs = list(matcher.get_opcodes())
        tag, i1, i2, j1, j2 = diffs[-1]
        if tag == 'insert' and (learner_sent[-1] in '。.!?' or j2 - j1 >= 10):
            target_sent = target_sent[:j1]
        elif tag == 'replace' and (j2 - j1) / (i2 - i1) >= 10:
            continue
        corr_lines.append(f'{target_sent}\n')
        incorr_lines.append(f'{learner_sent}\n')
        levels = edit_tagger(learner_sent, target_sent)
        edit_lines.extend(f'{lvl}\n' for lvl in levels)
    if not os.path.exists(part_output_dir):
        os.makedirs(part_output_dir)
    corr_path = os.path.join(part_output_dir, correct_file)
    incorr_path = os.path.join(part_output_dir, incorrect_file)
    edit_tags_path = os.path.join(part_output_dir, edit_tags_file)
    with open(corr_path, 'w', encoding='utf-8') as f:
        f.writelines(corr_lines)
    with open(incorr_path, 'w', encoding='utf-8') as f:
        f.writelines(incorr_lines)
    with open(edit_tags_path, 'wb') as f:
        edit_lines_bytes = ''.join(edit_lines).encode('utf-8')
        edit_lines_compressed = bz2.compress(edit_lines_bytes)
        f.write(edit_lines_compressed)
    print(f'Processed {len(corr_lines)} sentences, ' \
          f'{len(edit_lines)} edit-tagged sentences to {part_output_dir}')
    return len(corr_lines), len(edit_lines)


def preprocess_lang8(source_file, output_dir):
    """Generate edit-tagged sentence corpus from Lang8 corpus."""
    lines = []
    with open(source_file, encoding='utf-8') as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        row = json.loads(invalid_bytes_re.sub('', line))
        if row[2] == 'Japanese':
            rows.append(row)
    r = 8192
    rows_parts = [(rows[i:i + r], os.path.join(output_dir, str((i//r)+1)))
                  for i in range(0, len(rows), r)]
    print(f'Loaded {len(rows)} Japanese entries into {len(rows_parts)} parts')
    pool = Pool()
    pool_results = pool.map(preprocess_lang8_part, rows_parts)
    n_sents = sum(res[0] for res in pool_results)
    n_edit_sents = sum(res[1] for res in pool_results)
    print(f'Processed {n_sents} sentences and ' \
          f'{n_edit_sents} edit-tagged sentences.')


def main(args):
    preprocess_lang8(args.source, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',
                        help='Path to Lang8 corpus file',
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Path to output directory',
                        required=True)
    args = parser.parse_args()
    main(args)