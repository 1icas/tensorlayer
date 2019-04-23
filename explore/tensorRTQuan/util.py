#-127 - 127
import numpy as np
from scipy import stats

import math, copy

QUANTIZE_NUM = 127
BINS_NUM = 127


#quantize the output(after the activation, also we assume use the relu activation, so 
# we only consider the positive axis)
class QuantizeOutput:
    def __init__(self, default_bins, max_bins, need_quantize_output_count):
        self.th = 0
        self.default_bins = default_bins
        self.max_bins = max_bins
        self.need_quantize_output_count = need_quantize_output_count
        self.hist_counts = np.zeros((self.need_quantize_output_count, max_bins))
        self.bin_edges = np.zeros((self.need_quantize_output_count, max_bins+1))
        self.th_s = np.zeros((self.need_quantize_output_count))

    def find_max_output(self, *args):
        for idx, output in enumerate(args):
            th = np.max(output)
            if th > self.th_s[idx]:
                self.th_s[idx] = th

    def histogram(self, *args):
        for idx, output in enumerate(args):
            hist, bin_edge = np.histogram(output, bins=self.max_bins, range=(0, self.th_s[idx]))
            self.hist_counts[idx] += hist
            self.bin_edges[idx] = bin_edge

    #only support 2 dim or 4 dim now
    #and now we use the relu activation so we consider the positive asix only
    def weight_scale(self, w): 
        dims = w.ndim
        if dims != 2 and dims != 4:
            print("we only support 2 or 4 dims weight now")
            return
        if dims == 4:
            w = w.swapaxes(3, 0)
        elif dims == 2:
            w = w.swapaxes(1, 0)
        n = w.shape[0]
        scales = np.zeros(n)
        for idx in range(n):
            max_value = w[idx].max()
            min_value = abs(w[idx].min())
            value = max_value if min_value < max_value else min_value
            scale = 127 / value
            scales[idx] = scale
            w[idx] = np.floor(w[idx] * scale + 0.5)
            w[idx][w[idx] > 127] = 127
            # print(w[idx].min())
            # print(w[idx].max())
        if dims == 4:
            w = w.swapaxes(3, 0)
        elif dims == 2:
            w = w.swapaxes(1, 0)    
        return w, scales

    def output_value_scale(self):
        size = len(self.hist_counts)
        output_scale = []
        #output_scale = np.zeros(size)
        for idx in range(size):
            th, index = self.find_best_bin(self.hist_counts[idx])
            output_scale.append(np.array([127 / self.bin_edges[idx][index+2]]))
            print(self.bin_edges[idx][index+2])
        return output_scale
    
    #find minimum KL_divergence
    def find_best_bin(self, hist_count):
        th = 1000
        index = 0
        hist_count = hist_count[1:]
        size = len(hist_count)
        for target_bin in range(self.default_bins, size):
            q = np.zeros(target_bin)
            bin = hist_count[:target_bin]
            q[:target_bin] = hist_count[:target_bin]
            q[-1] = np.sum(hist_count[target_bin:])
            n = np.zeros(self.default_bins, dtype=np.int32)
            ratio = (target_bin) // self.default_bins
            is_not_zero = (q != 0).astype(np.int32)
            for idx in range(self.default_bins-1):
                start = idx * ratio
                end = start + ratio
                n[idx] = np.sum(bin[start:end])
            n[-1] = np.sum(bin[end:])
            p = np.zeros(target_bin, dtype=np.float64)
            for idx in range(self.default_bins-1):
                start = idx * ratio
                end = start + ratio
                count = np.sum(is_not_zero[start:end])
                if count == 0:
                    p[start:end] = 0
                else:
                    p[start:end] = n[idx] / float(count)
            count = np.sum(is_not_zero[end:])
            if count == 0:
                p[end:] = 0
            else:
                p[end:] = n[-1] / float(count)
            p[q == 0] = 0.0001
            q[q == 0] = 0.0001
            v = stats.entropy(q, p)
            if v < th:
                th = v
                index = target_bin
        return th, index
       

    def threshold_distribution1(self, distribution, target_bin=128):
        """
        Return the best threshold value. 
        Ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
        Args:
            distribution: list, activations has been processed by histogram and normalize,size is 2048
            target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
        Returns:
            target_threshold: int, num of bin with the minimum KL 
        """   
        distribution = distribution[1:]
        length = distribution.size
        threshold_sum = sum(distribution[target_bin:])
        kl_divergence = np.zeros(length - target_bin)
        thh = 1000
        index = 0
        for threshold in range(target_bin, length):
            sliced_nd_hist = copy.deepcopy(distribution[:threshold])

            # generate reference distribution p
            p = sliced_nd_hist.copy()
            p[threshold-1] += threshold_sum
            threshold_sum = threshold_sum - distribution[threshold]

            # is_nonzeros[k] indicates whether hist[k] is nonzero
            is_nonzeros = (p != 0).astype(np.int64)
            # 
            quantized_bins = np.zeros(target_bin, dtype=np.int64)
            # calculate how many bins should be merged to generate quantized distribution q
            num_merged_bins = sliced_nd_hist.size // target_bin
            
            # merge hist into num_quantized_bins bins
            for j in range(target_bin):
                start = j * num_merged_bins
                stop = start + num_merged_bins
                quantized_bins[j] = sliced_nd_hist[start:stop].sum()
            quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()
            
            # expand quantized_bins into p.size bins
            q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
            for j in range(target_bin):
                start = j * num_merged_bins
                if j == target_bin - 1:
                    stop = -1
                else:
                    stop = start + num_merged_bins
                norm = is_nonzeros[start:stop].sum()
                if norm != 0:
                    q[start:stop] = float(quantized_bins[j]) / float(norm)
            #q[p == 0] = 0
            # p = _smooth_distribution(p) # with some bugs, need to fix
            # q = _smooth_distribution(q)
            p[p == 0] = 0.0001
            q[q == 0] = 0.0001
            
            # calculate kl_divergence between q and p
            d = stats.entropy(p, q)
            if d < thh:
                thh = d
                index = threshold
        # min_kl_divergence = np.argmin(kl_divergence)
        # print(np.min(kl_divergence))
        # threshold_value = min_kl_divergence + target_bin
        print(thh)
        print(index)
        # return threshold_value



    # def threshold_distribution1(self, distribution, target_bin=128):
    #     """
    #         Returen the best cut off num of bin 
    #         Args:
    #             distribution: list, activations has been processed by histogram and normalize,size is 2048
    #             target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
    #         Returns:
    #             target_threshold: int, num of bin with the minimum KL 
    #     """   
    #     target_threshold = target_bin
    #     min_kl_divergence = 1000
    #     length = distribution.size

    #     quantize_distribution = np.zeros(target_bin)

    #     threshold_sum = 0.0
    #     threshold_sum = sum(distribution[target_bin:])

    #     for threshold in range(target_bin, length):
    #         t_distribution = copy.deepcopy(distribution[:threshold])
    #         t_distribution[threshold-1] = t_distribution[threshold-1] + threshold_sum
    #         threshold_sum = threshold_sum - distribution[threshold]

    #         # ************************ threshold  ************************
    #         quantize_distribution = np.zeros(target_bin)
    #         num_per_bin = threshold / target_bin
    #         for i in range(0, target_bin):
    #             start = i * num_per_bin
    #             end = start + num_per_bin

    #             left_upper = (int)(math.ceil(start))
    #             if(left_upper > start):
    #                 left_scale = left_upper - start
    #                 quantize_distribution[i] += left_scale * distribution[left_upper - 1]

    #             right_lower = (int)(math.floor(end))
    #             if (right_lower < end):
    #                 right_scale = end - right_lower
    #                 quantize_distribution[i] += right_scale * distribution[right_lower]

    #             for j in range(left_upper, right_lower):
    #                 quantize_distribution[i] += distribution[j]
    #         # ************************ threshold ************************

    #         # ************************ quantzie ************************
    #         expand_distribution = np.zeros(threshold, dtype=np.float32)

    #         for i in range(0, target_bin):
    #             start = i * num_per_bin
    #             end = start + num_per_bin

    #             count = 0

    #             left_upper = (int)(math.ceil(start))
    #             left_scale = 0.0
    #             if (left_upper > start):
    #                 left_scale = left_upper - start
    #                 if (distribution[left_upper - 1] != 0):
    #                     count += left_scale

    #             right_lower = (int)(math.floor(end))
    #             right_scale = 0.0
    #             if (right_lower < end):
    #                 right_scale = end - right_lower
    #                 if (distribution[right_lower] != 0):
    #                     count += right_scale

    #             for j in range(left_upper, right_lower):
    #                 if (distribution[j] != 0):
    #                     count = count + 1

    #             expand_value = quantize_distribution[i] / count

    #             if (left_upper > start):
    #                 if (distribution[left_upper - 1] != 0):
    #                     expand_distribution[left_upper - 1] += expand_value * left_scale
    #             if (right_lower < end):
    #                 if (distribution[right_lower] != 0):
    #                     expand_distribution[right_lower] += expand_value * right_scale
    #             for j in range(left_upper, right_lower):
    #                 if (distribution[j] != 0):
    #                     expand_distribution[j] += expand_value
    #         # ************************ quantzie ************************

    #         kl_divergence = stats.entropy(t_distribution, expand_distribution)

    #         if kl_divergence < min_kl_divergence:
    #             min_kl_divergence = kl_divergence
    #             target_threshold = threshold
    #     print(min_kl_divergence)
    #     return target_threshold

