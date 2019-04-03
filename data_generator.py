import random


class SequenceData(object):
    """
        generate sequence data.
        type 0: linear data , i.e. [0, 1, 2, 3, ...]
        type 1: random data, i.e. [1, 3, 7, 10, ...]
        max_seq_len: maximum length of sequence. if sequence length is less than
                    max_seq_len , fill will 0
    """
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=100):
        self.data = []
        self.labels = []
        self.seq_len = []
        self.batch_id = 0
        for i in range(n_samples):
            seq_len = random.randint(min_seq_len, max_seq_len)
            self.seq_len.append(seq_len)
            if random.random() < 0.5:
                rand_start = random.randint(0, max_value - seq_len)
                s = [[float(i) / max_value] for i in
                     range(rand_start, rand_start + seq_len)]
                s += [[0.0] for _ in range(max_seq_len - seq_len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                s = [[float(random.randint(0, max_value)) / max_value] for _
                     in range(seq_len)]
                s += [[0.0] for _ in range(max_seq_len - seq_len)]
                self.data.append(s)
                self.labels.append([0., 1.])

    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_size_id = min(self.batch_id + batch_size, len(self.data))
        batch_data = (self.data[self.batch_id:batch_size_id])
        batch_labels = (self.labels[self.batch_id:batch_size_id])
        batch_seq_len = (self.seq_len[self.batch_id:batch_size_id])
        self.batch_id = min(self.batch_id+batch_size, len(self.data))
        return batch_data, batch_labels, batch_seq_len


if __name__ == "__main__":
    runData = SequenceData()
    data, labels, seq_lens = runData.next(10)
    print(data)
    print(labels)
    print(seq_lens)
