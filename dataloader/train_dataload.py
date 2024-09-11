import os
import csv
import cv2
import numpy as np
from torch.utils.data import Dataset


class load_train(Dataset):
    # 1. initialize
    def __init__(self, root, csv_load, mode, args):  # root is the path,

        super(load_train, self).__init__()  # initialize
        self.root = root
        self.csv_load = csv_load
        self.mode = mode
        self.interval = [1]
        self.clip_len = args.clip_len
        self.resize = args.resize
        self.images, self.labels = self.load_csv(os.path.join(self.csv_load, self.mode + '.csv'), args)

    def load_csv(self, filename, args):
        # 1.Create a CSV file
        csv_load = os.path.join(self.root, filename)
        if not os.path.exists(os.path.join(self.csv_load, self.mode + '.csv')):
            seqs = []
            # for load in os.listdir(self.root):#Use enriched datasets
            #     data_load = os.path.join(self.root, load)
            #     video_dir = self.load_files(data_load, self.mode)
            video_dir = self.load_files(self.root, self.mode)
            for inter in self.interval:
                for video in video_dir:
                    get_len = len(os.listdir(video)) - inter * self.clip_len
                    if get_len > 0:
                        label = int(video.split(os.sep)[-2])
                        seqs = self.load_video_sequence(video, inter, label, seqs)
                    else:
                        pass
            with open(csv_load, mode='w', newline='') as f:
                writer = csv.writer(f)
                for seq in seqs:
                    # Gets the label based on the file name
                    writer.writerow(seq)
                print('writen into csv file:', csv_load)

        # 2.Read the image sequence and label from the CSV, and if the CSV code does not exist, perform this step directly
        images, labels = [], []
        print('open(csv_load)')
        with open(csv_load) as f:
            reader = csv.reader(f)
            for row in reader:
                video, inter, time_index, label = row
                label = int(label)
                img_seqs = self.read_seq(video, inter, time_index)
                images.append(img_seqs)
                labels.append(label)
        assert len(images) == len(labels)  # Ensure consistency in the number of data and labels
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # Create an index for images and tags
        seqs, label = self.images[idx], self.labels[idx]
        if self.mode == 'train':
            seqs = self.randomflip(seqs)
        seqs = self.normalize(seqs)
        seqs = self.to_tensor(seqs)
        return seqs, label

    def normalize(self, seqs):
        for i, frame in enumerate(seqs):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            seqs[i] = frame

        return seqs

    def randomflip(self, buffer):

        """
        Horizontally flip the given image and ground truth randomly with a probability of 0.5.
        """
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def read_seq(self, video, inter, time_index):
        time_index = int(time_index)
        inter = int(inter)
        frames = sorted([os.path.join(video, img) for img in os.listdir(video)])
        frame_count = self.clip_len
        img_seqs = np.empty((frame_count, self.resize, self.resize, 3), np.dtype('float32'))
        for i in range(0, frame_count):
            frame_resize = cv2.resize(cv2.imread(frames[time_index]), (self.resize, self.resize))
            
            ####
            '''
            img = np.uint8(frame_resize)

            imgr = img[:,:,0]
            imgg = img[:,:,1]
            imgb = img[:,:,2]
        
            claher = cv2.createCLAHE(clipLimit=3, tileGridSize=(10,18))
            claheg = cv2.createCLAHE(clipLimit=2, tileGridSize=(10,18))
            claheb = cv2.createCLAHE(clipLimit=1, tileGridSize=(10,18))
            cllr = claher.apply(imgr)
            cllg = claheg.apply(imgg)
            cllb = claheb.apply(imgb)
        
            frame_resize = np.dstack((cllr,cllg,cllb))
            '''
            ####
            
            frame = np.array(frame_resize).astype(np.float64)
            img_seqs[i] = frame
            time_index = time_index + inter
        return img_seqs

    def load_video_sequence(self, video, inter, label, seqs):
        clip_len = self.clip_len
        frames = sorted([os.path.join(video, img) for img in os.listdir(video)])
        max_len = len(frames) - clip_len * inter
        time_index = 0
        '''
        if (len(frames) - clip_len * inter)>0:
            time_index = np.random.randint(0,clip_len * inter)
        else :
            time_index = 0
        time_index = int(time_index)
        '''
        """
        Use an augmented dataset with partial sampling
        """
        while time_index < max_len:
            seqs.append([video, inter, time_index, label])
            time_index = time_index + clip_len * inter
            #time_index = time_index+1
            #time_index = time_index+clip_len//2

        else:
            pass
        # time_index = np.random.randint(len(frames) - clip_len * inter)
        # time_index = int(time_index)
        # """
        # Using an augmented dataset, partial sampling,
        # """
        # if time_index < max_len:
        #     seqs.append([video, inter, time_index, label])
        #     # time_index = int(time_index + clip_len * inter)
        #
        # else:
        #     pass

        return seqs

    # 1.read path
    def load_files(self, dir, mode):
        video_dir = []
        members = []
        if mode == 'train':
            members = ['train', 'dev']
        elif mode == 'test':
            members = ['test']
        else:
            print("path error")
        for member in members:
            print(member)
            load = os.path.join(dir, member)  # Use enriched datasets
            if os.path.exists(load):
                print('load', load)
                for label in os.listdir(load):
                    label_load = os.path.join(load, label)
                    for video in os.listdir(label_load):
                        video_load = os.path.join(label_load, video)
                        video_dir.append(video_load)
        # print(video_dir)
        return video_dir


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse

    Path = "data/AVEC2013/"
    load = "LMTformer/csv_load/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--resize', default=128, type=int, metavar='N')
    parser.add_argument('--clip_len', default=24, type=int, metavar='N')
    parser.add_argument('--loads', default=[100, 200, 300, 400], type=str)
    # parser.add_argument('--beta', type=flo at, default=0.9)  
    args = parser.parse_args()
    train_data = load_train(root=Path, csv_load=load, mode='test', args=args)
    # (self, root, resize, csv_load, mode, clip_len)
    # print(train_data)
    train_loader = DataLoader(train_data, batch_size=25, shuffle=True)
    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print('input:', inputs.size())
        print('labels:', labels)
        if i == 1:
            break
