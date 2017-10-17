import sys

def main(manifest_filepath):
    with open(manifest_filepath) as f:
        ids = f.readlines()
    ids = [x.strip().split(',') for x in ids]
    for i in ids:
        txt_file = i[2]
        with open(txt_file) as t:
            for l in t:
                print(l.strip())

if __name__ == '__main__':
    main(sys.argv[1])
