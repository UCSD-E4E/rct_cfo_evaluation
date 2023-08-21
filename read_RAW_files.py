import numpy as np
import matplotlib.pyplot as plt
import struct
import os
from smb_unzip.smb_unzip import smb_unzip
import datetime
from pathlib import Path
import argparse; parser = argparse.ArgumentParser(description='RAW_DATA folder path')

def read_single_RAW(dataFilePath: Path) -> list:
    '''
        read a single RAW_DATA file
    '''
    nSamples = int(dataFilePath.stat().st_size / 4)
    signal_raw = np.zeros(nSamples, dtype=np.complex128)
    with open(file=dataFilePath, mode='rb') as dataFile:
        for i in range(nSamples):
            sampleBytes = dataFile.read(4)
            re, im = struct.unpack("<2h", sampleBytes)
            signal_raw[i] = float(re) / 0x7fff + float(im) * 1j / 0x7fff
    # t_raw = np.arange(0, nSamples / f_s, 1/f_s)
    return signal_raw


def read_raw_files(RAW_DATA_path: Path, countLimit: int) -> list[list]:
    '''
        read countLimit number of RAW_DATA files from the specified RAW_DATA_path
    '''

    print(f'reading up to {countLimit} RAW_DATA files in {RAW_DATA_path}')
    RAW_signal_collection = []
    count = 0
    for child in RAW_DATA_path.iterdir():
        if child.parts[-1].startswith('RAW_DATA_') and count < countLimit:
            count += 1
            RAW_signal_collection.append(read_single_RAW(child))
    return RAW_signal_collection


# if __name__ == '__main__':
#     parser.add_argument(
#         '--path', help='path of RAW_DATA folder', required=True, type=Path)
#     parser.add_argument(
#         '--count', help='number of RAW_DATA files to read', required=True, type=int)

#     args = parser.parse_args()

#     RAW_signal_collection = read_raw_files(args.path, args.count)