'''
# Generate image with random string & font & size
1. select font
1. select size
1. select chars 2 ~ 10
1. pick random position
1. check fill_image_ratio

'''

import random
import copy
from string import ascii_lowercase, ascii_uppercase
from os import listdir, path, makedirs
from os.path import isfile, isdir, join, basename

begin = 0xac00
end = 0xd7a3

class CH_Dataset():
    def __init__(self, font_path=None):
        self.char_list = [chr(begin + idx) for idx in range(end - begin + 1)] \
                         + [x for x in (ascii_lowercase + ascii_uppercase)] + [str(x) for x in range(10)] + [x for x in
                                                                                                             '~!@#$%^&*()_+-=<>?,.;:[]{}|']

        self.char_list = self.char_list[:30]
        self.font_list = [join(font_path, f) for f in listdir(font_path) if
                          isfile(join(font_path, f)) and f.find('.DS_Store') == -1]

        self.font_sizes = [10] * 2 + [20] * 5 + [30] * 3 + [50] * 2 + [100]

        self.counter = 0
        self.gen_counter = 0
        self.fill_ratio = 0

        self.font_size_tuples = None
        self.char_list_list = None
        self.counter_list = None

    def isStop(self):
        self.counter = self.counter + 1
        if self.counter > 50:
            return True
        else:
            return False

    def hasFill(self):
        self.fill_ratio += 2

        if self.fill_ratio > 50:
            return True
        else:
            return False

    def getFont(self):
        if self.font_list is not None and len(self.font_list) > 0 and len(self.font_sizes) > 0:
            self.gen_counter += 1
            return self.font_list[self.gen_counter % len(self.font_list)], self.font_sizes[
                self.gen_counter % len(self.font_sizes)]
        else:
            return None, None

    def getChars(self):
        print('self.counter_list', self.counter_list)

        # pick a random char list
        idx = random.randrange(len(self.char_list_list))

        # split first k chars(1~10)
        k_chars = random.randrange(10) + 1

        # get fist k chars
        current_char_idx = self.counter_list[idx]

        # make sure target_idx <= len(self.char_list_list[idx])
        target_idx = current_char_idx + k_chars if current_char_idx + k_chars < len(self.char_list_list[idx]) else len(
            self.char_list_list[idx])

        print(idx, current_char_idx, target_idx, self.char_list_list[idx][current_char_idx:target_idx])

        # pick chars (current_char_idx:target_idx)
        sample_chars = self.char_list_list[idx][current_char_idx:target_idx]
        self.counter_list[idx] = target_idx

        print('self.counter_list', self.counter_list)

    def generate(self, n_char=10, ratio=0.5, width=800, height=600):
        '''
        generate single image sample containing multiple chars

        :param n_char: number of char per font & size
        :param ratio: fill area ratio of image with character region
        :param width: width of image
        :param height: height of image
        :return:
        '''
        pass

        if self.font_size_tuples is None or self.char_list_list is None or self.counter_list is None:
            return None

        self.getChars()
        #         while self.hasFill() is False:
        #             font = self.sampleText()
        #             print(font)

    def generateSamples(self, n_char=10, fill_ratio=0.5, width=800, height=600):
        '''
        generate samples:
            1. generate (font, size) tuple
            1. create N char list(N = # of (font, size) tuple)
            1. shuffle each char list

        :param n_char: number of char per font & size
        :param ratio: fill area ratio of image with character region
        :param width: width of image
        :param height: height of image

        :return:
        '''

        self.font_size_tuples = [(f, s) for f in self.font_list for s in self.font_sizes][:5]
        self.char_list_list = [copy.deepcopy(self.char_list) for _ in range(len(self.font_size_tuples))]
        [random.shuffle(ch_list) for ch_list in self.char_list_list]
        self.counter_list = [0] * len(self.char_list_list)

        results = list()
        while self.isStop() is False:
            gen_img = self.generate()
            if gen_img:
                results.append(gen_img)
        return results


chd = CH_Dataset(font_path='/Users/ali.jeon/Desktop/fonts')
chd.generateSamples()
