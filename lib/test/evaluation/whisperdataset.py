import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

class whisperDataset_nir(BaseDataset):

    def __init__(self):
        super().__init__()

        self.base_path = self.env_settings.whisper_nir

        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])


        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'whisper', ground_truth_rect[init_omit:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):

        folder_path_false = self.base_path
        # 使用列表推导式获取子目录的文件夹名字
        name = [nam for nam in os.listdir(folder_path_false) if os.path.isdir(os.path.join(folder_path_false, nam))]
        number = []
        for i in name:
            i = i
            count = 0
            for file in os.listdir(folder_path_false + "/" + i):
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".bmp") or file.endswith(".tif"):
                    count += 1
            number.append(count)
        print(number)
        sequence_info_list = []
        for i, j in zip(name, number):
            seq = {"name": i, "path":  i, "startFrame": 1, "endFrame": j, "nz": 4, "ext": "png",
                   "anno_path": i + "/groundtruth_rect.txt", "object_class": i}
            sequence_info_list.append(seq)


        return sequence_info_list


class whisperDataset_vis(BaseDataset):

    def __init__(self):
        super().__init__()

        self.base_path = self.env_settings.whisper_vis

        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])


        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'whisper', ground_truth_rect[init_omit:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):

        folder_path_false = self.base_path
        # 使用列表推导式获取子目录的文件夹名字
        name = [nam for nam in os.listdir(folder_path_false) if os.path.isdir(os.path.join(folder_path_false, nam))]
        number = []
        for i in name:
            i = i
            count = 0
            for file in os.listdir(folder_path_false + "/" + i):
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".bmp"):
                    count += 1
            number.append(count)
        print(number)
        sequence_info_list = []
        for i, j in zip(name, number):
            seq = {"name": i, "path": i , "startFrame": 1, "endFrame": j, "nz": 4, "ext": "png",
                   "anno_path": i  + "/groundtruth_rect.txt", "object_class": i}
            sequence_info_list.append(seq)


        return sequence_info_list




class whisperDataset_rednir(BaseDataset):

    def __init__(self):
        super().__init__()

        self.base_path = self.env_settings.whisper_rednir

        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])


        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'whisper', ground_truth_rect[init_omit:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):

        folder_path_false = self.base_path
        # 使用列表推导式获取子目录的文件夹名字
        name = [nam for nam in os.listdir(folder_path_false) if os.path.isdir(os.path.join(folder_path_false, nam))]
        number = []
        for i in name:
            i = i
            count = 0
            for file in os.listdir(folder_path_false + "/" + i):
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".bmp"):
                    count += 1
            number.append(count)
        print(number)
        sequence_info_list = []
        for i, j in zip(name, number):
            seq = {"name": i, "path": i, "startFrame": 1, "endFrame": j, "nz": 4, "ext": "png",
                   "anno_path": i + "/groundtruth_rect.txt", "object_class": i}
            sequence_info_list.append(seq)


        return sequence_info_list