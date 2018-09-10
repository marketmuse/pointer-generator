import os
import argparse

def clean_sentence(sentence):

	return cleaned


def main(args):

	if args.out_path is None:
		out_path = args.data_path.split('.')[0]+'_cleaned.tsv'
	else:
		out_path = args.out_path
	with open(args.data_path) as r, open(out_path, 'w') as w:

		for entry in r:
			splits = r.strip().split('\t')
			s1, s2 = splits[0], splits[1]

			for i, s in enumerate([s1, s2]):
				splits[i] = clean_sentence(s)

			w.write('\t'.join(splits)+'\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path', help="path of raw data file to clean")
	parser.add_argument('--out_path', help="path to write clean data file", default=None)
	main(args)