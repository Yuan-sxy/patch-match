import numpy as np
import cv2 as cv
import time
from PIL import Image

debug = True
detail = True


class PatchMatch:
    def __init__(self, src, ref, p, it, a):
        self.src_array = np.array(src)
        self.ref_array = np.array(ref)
        self.patch_size = p
        self.iteration = it

        self.height = np.size(self.src_array, 0)  # np,size(,0) returns row number
        self.width = np.size(self.src_array, 1)  # np.size(,1) returns column number
        self.ref_height = np.size(self.ref_array, 0)
        self.ref_width = np.size(self.ref_array, 1)
        if debug:
            print('src_array:height-%d, width-%d' % (self.height, self.width))
            print('ref_array:height-%d, width-%d' % (self.ref_height, self.ref_width))
        # right and bottom need patch_size length padding
        self.padding_matrix = np.ones([self.height + self.patch_size, self.width + self.patch_size, 3]) * np.nan
        self.padding_matrix[:self.height, :self.width, :] = self.src_array

        self.offset = self.get_initial_offset()
        self.dist = self.get_initial_dist()
        self.alpha = a
        if debug:
            print("finish initial!")
            # print('offset matrix:', self.offset)
            # print('dist matrix:', self.dist)

    def cal_distance(self, patch_a, patch_b):
        tmp = patch_b - patch_a
        # Only count those pixels which are not np.nan
        num = np.sum(1 - np.int32(np.isnan(tmp)))
        # np.nan_to_num use 0 replacing nan
        dist = np.sum(np.square(np.nan_to_num(tmp))) / num
        return dist

    def get_initial_offset(self):
        offset = np.zeros([self.height, self.width], dtype=object)
        for i in range(0, self.height):
            for j in range(0, self.width):
                h = np.random.randint(0, self.ref_height - self.patch_size)
                w = np.random.randint(0, self.ref_width - self.patch_size)
                offset[i][j] = np.array([h, w], dtype=np.int32)
        if debug:
            print("get initial offset matrix!")
            # offset.tofile('initial_offset.csv',sep='|')
        return offset

    def get_initial_dist(self):
        dist = np.zeros([self.height, self.width])
        for i in range(0, self.height):
            for j in range(0, self.width):
                patch_a = self.padding_matrix[i:i + self.patch_size, j:j + self.patch_size, :]

                h = self.offset[i][j][0]
                w = self.offset[i][j][1]
                patch_b = self.ref_array[h:h + self.patch_size, w:w + self.patch_size, :]

                dist[i][j] = self.cal_distance(patch_a, patch_b)

        if debug:
            print("get initial dist matrix!")
        # dist.tofile('initial_dist.csv',sep='|')
        return dist

    def propagation(self, i, j, inverse):

        if inverse:
            d_bottom = self.dist[min(i + 1, self.height - 1)][j]
            d_right = self.dist[i][min(j + 1, self.width - 1)]
            d_curr = self.dist[i][j]
            idx = np.argmin(np.array([d_right, d_bottom, d_curr]))
            if idx == 0:
                self.offset[i][j] = self.offset[i][min(j + 1, self.width - 1)]

                patch_a = self.padding_matrix[i:i + self.patch_size, j:j + self.patch_size, :]

                h = self.offset[i][j][0]
                w = self.offset[i][j][1]
                patch_b = self.ref_array[h:h + self.patch_size, w:w + self.patch_size, :]

                self.dist[i][j] = self.cal_distance(patch_a, patch_b)
            elif idx == 1:
                self.offset[i][j] = self.offset[min(i + 1, self.height - 1)][j]

                patch_a = self.padding_matrix[i:i + self.patch_size, j:j + self.patch_size, :]

                h = self.offset[i][j][0]
                w = self.offset[i][j][1]
                patch_b = self.ref_array[h:h + self.patch_size, w:w + self.patch_size, :]

                self.dist[i][j] = self.cal_distance(patch_a, patch_b)
        else:
            d_top = self.dist[max(i - 1, 0)][j]
            d_left = self.dist[i][max(j - 1, 0)]
            d_curr = self.dist[i][j]
            idx = np.argmin(np.array([d_left, d_top, d_curr]))
            if idx == 0:
                self.offset[i][j] = self.offset[i][max(j - 1, 0)]

                patch_a = self.padding_matrix[i:i + self.patch_size, j:j + self.patch_size, :]

                h = self.offset[i][j][0]
                w = self.offset[i][j][1]
                patch_b = self.ref_array[h:h + self.patch_size, w:w + self.patch_size, :]

                self.dist[i][j] = self.cal_distance(patch_a, patch_b)
            elif idx == 1:
                self.offset[i][j] = self.offset[max(i - 1, 0)][j]

                patch_a = self.padding_matrix[i:i + self.patch_size, j:j + self.patch_size, :]

                h = self.offset[i][j][0]
                w = self.offset[i][j][1]
                patch_b = self.ref_array[h:h + self.patch_size, w:w + self.patch_size, :]

                self.dist[i][j] = self.cal_distance(patch_a, patch_b)

    def random_search(self, i, j):
        search_stride_h = self.ref_height
        search_stride_w = self.ref_width
        x = self.offset[i][j][0]
        y = self.offset[i][j][1]
        i = 3
        while search_stride_h > 1 and search_stride_w > 1:
            h_start = max(0, x - search_stride_h)
            h_end = min(self.ref_height - self.patch_size, x + search_stride_h)

            w_start = max(0, y - search_stride_w)
            w_end = min(self.ref_width - self.patch_size, y + search_stride_w)

            h = np.random.randint(h_start, h_end)
            w = np.random.randint(w_start, w_end)

            patch_a = self.padding_matrix[i:i + self.patch_size, j:j + self.patch_size, :]
            patch_b = self.ref_array[h:h + self.patch_size, w:w + self.patch_size, :]
            tmp = self.cal_distance(patch_a, patch_b)
            if self.dist[i][j] > tmp:
                self.dist[i][j] = tmp
                self.offset[i][j] = np.array([h, w])

            i += 1
            search_stride_h = self.ref_height * (self.alpha ** i)
            search_stride_w = self.ref_width * (self.alpha ** i)

    def nearest_neighbour_field(self):
        for it in range(0, self.iteration):
            if debug:
                print("begin %d iteration!" % it)
                on_begin = time.time()
            flag = (it % 2 == 0)
            if flag:
                for i in range(0, self.height):
                    for j in range(0, self.width):
                        self.propagation(i, j, False)
                        self.random_search(i, j)
            else:
                for i in range(self.height - 1, -1, -1):
                    for j in range(self.width - 1, -1, -1):
                        self.propagation(i, j, True)
                        self.random_search(i, j)
            if debug:
                on_end = time.time()
                print("finish %d iteration in %d second!" % (it, on_end - on_begin))
                # self.offset.tofile('initial_offset_%d_interation.csv'%it,sep='|')
            if detail:
                tmp = np.zeros_like(self.src_array)
                for i in range(0, self.height):
                    for j in range(0, self.width):
                        h = self.offset[i][j][0]
                        w = self.offset[i][j][1]
                        tmp[i, j, :] = self.ref_array[h, w, :]
                tmp_img = Image.fromarray(tmp)
                tmp_img.save('iteration%d.jpg' % it)

    def reconstruct(self):
        if debug:
            print("begin reconstruct!")
        result = np.zeros_like(self.src_array)
        for i in range(0, self.height):
            for j in range(0, self.width):
                h = self.offset[i][j][0]
                w = self.offset[i][j][1]
                result[i, j, :] = self.ref_array[h, w, :]
        # self.result.tofile('result_matrix.csv',sep='|')
        result_img = Image.fromarray(result)
        result_img.show()
        result_img.sa
        print('finish reconstruction!')


if __name__ == "__main__":
    # read image and change it to numpy array
    src_img = Image.open("cup_a.jpg")
    ref_img = Image.open("cup_b.jpg")
    # src, ref has 3 dimensions [height,width,channels]

    # set patch size and iteration
    patch_size = 3
    iteration = 5

    patch_match = PatchMatch(src_img, ref_img, patch_size, iteration, 0.5)

    # count exec time
    start_time = time.time()
    patch_match.nearest_neighbour_field()
    patch_match.reconstruct()

    end_time = time.time()
    print(end_time - start_time)
