import imageio
import cv2
import numpy as np
import os
import pickle
import re
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from PIL import Image
#from scipy.misc.pilutil import imresize
from torch.autograd import Variable

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from matplotlib import pyplot as plt
import argparse
import os

CATEGORIES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking"
]

# Dataset are divided according to the instruction at:
# http://www.nada.kth.se/cvap/actions/00sequences.txt
Training_ID = [11, 12, 13, 14, 15, 16, 17, 18]
Testing_ID = [19, 20, 21, 23, 24, 25, 1, 4]
Validation_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]

def raw_dataset(dataset="train"):
    if dataset == "train":
        ID = Training_ID
    elif dataset == "test":
        ID = Testing_ID
    else:
        ID = Validation_ID

    frames_index = parsing_sequence_file()

    data = []

    for category in CATEGORIES:
        # Get all files in current category's folder.
        folder_path = os.path.join("..", "dataset", category)
        filenames = sorted(os.listdir(folder_path))

        for filename in filenames:
            filepath = os.path.join("..", "dataset", category, filename)

            # Get id of person in this video.
            person_id = int(filename.split("_")[0][6:])
            if person_id not in ID:
                continue

            vid = imageio.get_reader(filepath, "ffmpeg")

            frames = []

            # Add each frame to correct list.
            for i, frame in enumerate(vid):
                # Boolean flag to check if current frame contains human.
                ok = False
                for seg in frames_index[filename]:
                    if i >= seg[0] and i <= seg[1]:
                        ok = True
                        break
                if not ok:
                    continue

                # Convert to grayscale.
                frame = Image.fromarray(np.array(frame))
                frame = frame.convert("L")
                frame = np.array(frame.getdata(),dtype=np.uint8).reshape((120, 160))
                frame = resize(frame, (60, 80))
                
                frames.append(frame)
            
            data.append({
                "filename": filename,
                "category": category,
                "frames": frames    
            })
            print(data)
    pickle.dump(data, open("data/%s.p" % dataset, "wb"))
    
def optflow_dataset(dataset="train"):
    if dataset == "train":
        ID = Training_ID
    elif dataset == "test":
        ID = Testing_ID
    else:
        ID = Validation_ID

    # Setup parameters for optical flow.
    farneback_params = dict(
        winsize=20, iterations=1,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
        pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)

    frames_index = parsing_sequence_file()

    data = []

    for category in CATEGORIES:
        # Get all files in current category's folder.
        folder_path = os.path.join("..", "dataset", category)
        filenames = sorted(os.listdir(folder_path))

        for filename in filenames:
            filepath = os.path.join("..", "dataset", category, filename)

            # Get id of person in this video.
            person_id = int(filename.split("_")[0][6:])
            if person_id not in ID:
                continue

            vid = imageio.get_reader(filepath, "ffmpeg")

            flow_x = []
            flow_y = []

            prev_frame = None
            # Add each frame to correct list.
            for i, frame in enumerate(vid):
                # Boolean flag to check if current frame contains human.
                ok = False
                for seg in frames_index[filename]:
                    if i >= seg[0] and i <= seg[1]:
                        ok = True
                        break
                if not ok:
                    continue

                # Convert to grayscale.
                frame = Image.fromarray(np.array(frame))
                frame = frame.convert("L")
                frame = np.array(frame.getdata(),dtype=np.uint8).reshape((120, 160))
                frame = resize(frame,(60, 80))

                if prev_frame is not None:
                    # Calculate optical flow.
                    flows = cv2.calcOpticalFlowFarneback(prev_frame, frame,**farneback_params)
                    subsampled_x = np.zeros((30, 40), dtype=np.float32)
                    subsampled_y = np.zeros((30, 40), dtype=np.float32)

                    for r in range(30):
                        for c in range(40):
                            subsampled_x[r, c] = flows[r*2, c*2, 0]
                            subsampled_y[r, c] = flows[r*2, c*2, 1]

                    flow_x.append(subsampled_x)
                    flow_y.append(subsampled_y)

                prev_frame = frame
                
            data.append({
                "filename": filename,
                "category": category,
                "flow_x": flow_x,
                "flow_y": flow_y    
            })

    pickle.dump(data, open("data/%s_flow.p" % dataset, "wb"))

def parsing_sequence_file():
    print("Parsing ../dataset/00sequences.txt")
    #listing= os.listdir('sequences.txt')
    # Read 00sequences.txt file.
    
    with open('../dataset/sequence.txt', 'r') as content_file:
        content = content_file.read()

    # Replace tab and newline character with space, then split file's content
    # into strings.
    content = re.sub("[\t\n]", " ", content).split()

    # Dictionary to keep ranges of frames with humans.
    # Example:
    # video "person01_boxing_d1": [(1, 95), (96, 185), (186, 245), (246, 360)].
    frames_index = {}

    # Current video that we are parsing.
    current_filename = ""

    for s in content:
        if s == "frames":
            # Ignore this token.
            continue
        elif s.find("-") >= 0:
            # This is the token we are looking for. e.g. 1-95.
            if s[len(s) - 1] == ',':
                # Remove comma.
                s = s[:-1]

            # Split into 2 numbers => [1, 95]
            idx = s.split("-")

            # Add to dictionary.
            if not current_filename in frames_index:
                frames_index[current_filename] = []
            frames_index[current_filename].append((int(idx[0]), int(idx[1])))
        else:
            # Parse next file.
            current_filename = s + "_uncomp.avi"

    return frames_index

CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5
}

class opticalflowdataset(Dataset):
    def __init__(self, directory, dataset="train", mean=None):
        self.instances, self.labels = self.read_dataset(directory, dataset)

        for i in range(len(self.instances)):
            self.instances[i]["frames"] = torch.from_numpy(
                self.instances[i]["frames"])
            self.instances[i]["flow_x"] = torch.from_numpy(
                self.instances[i]["flow_x"])
            self.instances[i]["flow_y"] = torch.from_numpy(
                self.instances[i]["flow_y"])

        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        sample = { 
            "instance": self.instances[idx], 
            "label": self.labels[idx] 
        }

        return sample

    def zero_center(self, mean):
        for i in range(len(self.instances)):
            self.instances[i]["frames"] -= float(mean["frames"])
            self.instances[i]["flow_x"] -= float(mean["flow_x"])
            self.instances[i]["flow_y"] -= float(mean["flow_y"])

    def read_dataset(self, directory, dataset="train", mean=None):
        if dataset == "train":
            frame_path = os.path.join(directory, "train.p")
            flow_path = os.path.join(directory, "train_flow.p")
        elif dataset == "test":
            frame_path = os.path.join(directory, "test.p")
            flow_path = os.path.join(directory, "test_flow.p")
        else:
            frame_path = os.path.join(directory, "validation.p")
            flow_path = os.path.join(directory, "validation_flow.p")

        video_frames = pickle.load(open(frame_path, "rb"))
        video_flows = pickle.load(open(flow_path, "rb"))

        instances = []
        labels = []

        mean_frames = 0
        mean_flow_x = 0
        mean_flow_y = 0

        for i_video in range(len(video_frames)):
            current_block_frame = []
            current_block_flow_x = []
            current_block_flow_y = []

            frames = video_frames[i_video]["frames"]
            flow_x = [0] + video_flows[i_video]["flow_x"]
            flow_y = [0] + video_flows[i_video]["flow_y"]

            for i_frame in range(len(frames)):
                current_block_frame.append(frames[i_frame])

                if i_frame % 15 > 0:
                    current_block_flow_x.append(flow_x[i_frame])
                    current_block_flow_y.append(flow_y[i_frame])

                if (i_frame + 1) % 15 == 0:
                    current_block_frame = np.array(
                        current_block_frame,
                        dtype=np.float32).reshape((1, 15, 60, 80))
                    current_block_flow_x = np.array(
                        current_block_flow_x,
                        dtype=np.float32).reshape((1, 14, 30, 40))
                    current_block_flow_y = np.array(
                        current_block_flow_y,
                        dtype=np.float32).reshape((1, 14, 30, 40))

                    mean_frames += np.mean(current_block_frame)
                    mean_flow_x += np.mean(current_block_flow_x)
                    mean_flow_y += np.mean(current_block_flow_y)

                    instances.append({
                        "frames": current_block_frame,
                        "flow_x": current_block_flow_x,
                        "flow_y": current_block_flow_y
                    })

                    labels.append(
                        CATEGORY_INDEX[video_frames[i_video]["category"]])

                    current_block_frame = []
                    current_block_flow_x = []
                    current_block_flow_y = []

        mean_frames /= len(instances)
        mean_flow_x /= len(instances)
        mean_flow_y /= len(instances)

        self.mean = {
            "frames": mean_frames,
            "flow_x": mean_flow_x,
            "flow_y": mean_flow_y
        }

        labels = np.array(labels, dtype=np.uint8)

        return instances, labels

class CNNOpticalFlow(nn.Module):
    def __init__(self):
        super(CNNOpticalFlow, self).__init__()

        self.conv1_frame = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(4, 5, 5)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout(0.5))
        self.conv2_frame = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(4, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))
        self.conv3_frame = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))

        self.conv1_flow_x = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout(0.5))
        self.conv2_flow_x = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))
        self.conv3_flow_x = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))

        self.conv1_flow_y = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout(0.5))
        self.conv2_flow_y = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))
        self.conv3_flow_y = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(0.5))

        self.fc1 = nn.Linear(3328, 128)
        self.dropfc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, frames, flow_x, flow_y):
        out_frames = self.conv1_frame(frames)
        out_frames = self.conv2_frame(out_frames)
        out_frames = self.conv3_frame(out_frames)
        out_frames = out_frames.view(out_frames.size(0), -1)

        out_flow_x = self.conv1_flow_x(flow_x)
        out_flow_x = self.conv2_flow_x(out_flow_x)
        out_flow_x = self.conv3_flow_x(out_flow_x)
        out_flow_x = out_flow_x.view(out_flow_x.size(0), -1)

        out_flow_y = self.conv1_flow_y(flow_y)
        out_flow_y = self.conv2_flow_y(out_flow_y)
        out_flow_y = self.conv3_flow_y(out_flow_y)
        out_flow_y = out_flow_y.view(out_flow_y.size(0), -1)

        out = torch.cat([out_frames, out_flow_x, out_flow_y], 1)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = self.dropfc1(out)
        out = self.fc2(out)

        return out

def get_outputs(model, instances, flow=False, use_cuda=False):

    if flow:
        frames = Variable(instances["frames"])
        flow_x = Variable(instances["flow_x"])
        flow_y = Variable(instances["flow_y"])

        if use_cuda:
            frames = frames.cuda()
            flow_x = flow_x.cuda()
            flow_y = flow_y.cuda()

        outputs = model(frames, flow_x, flow_y)

    else:
        instances = Variable(instances)
        if use_cuda:
            instances = instances.cuda()

        outputs = model(instances)

    return outputs

def evaluate(model, dataloader, flow=False, use_cuda=False):
    loss = 0
    correct = 0
    total = 0

    # Switch to evaluation mode.
    model.eval()

    for i, samples in enumerate(dataloader):
        outputs = get_outputs(model, samples["instance"], flow=flow,
                              use_cuda=use_cuda)
        
        labels = Variable(samples["label"])
        if use_cuda:
            labels = labels.cuda()

        loss += nn.CrossEntropyLoss(size_average=False)(outputs, labels).data

        score, predicted = torch.max(outputs, 1)
        correct += (labels.data == predicted.data).sum()
        
        total += labels.size(0)

    acc = correct / total
    loss /= total

    return loss, acc

def train(model, num_epochs, train_set, test_set, lr=1e-3, batch_size=32,
          start_epoch=1, log=10, checkpoint_path=None, validate=True,
          resume=False, flow=False, use_cuda=False):

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)

    # Must be sequential b/c this is used for evaluation.
    train_loader_sequential = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False)

    # Use Adam optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_accuracy = []
    test_accuracy = []
    # Record loss + accuracy.
    hist = []

    # Check if we are resuming training from a previous checkpoint.
    if resume:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, "model_epoch%d.chkpt" % (start_epoch - 1)))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        hist = checkpoint["hist"]

    if use_cuda:
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Switch to train mode.
        model.train()

        for i, samples in enumerate(train_loader):

            labels = Variable(samples["label"])
            if use_cuda:
                labels = labels.cuda()

            # Zero out gradient from previous iteration.
            optimizer.zero_grad()

            # Forward, backward, and optimize.
            outputs = get_outputs(model, samples["instance"], flow=flow,
                                  use_cuda=use_cuda)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % log == 0:
                print("epoch %d/%d, iteration %d/%d, loss: %s"
                      % (epoch, start_epoch + num_epochs - 1, i + 1,
                      len(train_set) // batch_size, loss.data))
        
        # Get overall loss & accuracy on training set.
        train_loss, train_acc = evaluate(model, train_loader_sequential,
                                         flow=flow, use_cuda=use_cuda)

        if validate:
            # Get overall loss & accuracy on test set.
            test_loss, test_acc = evaluate(model, test_loader, flow=flow,
                                         use_cuda=use_cuda)

            print("epoch %d/%d, train_loss = %s, train_acc = %s, "
                  "test_loss = %s, test_acc = %s"
                  % (epoch, start_epoch + num_epochs - 1,
                  train_loss, train_acc, test_loss, test_acc))
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)
            hist.append({
                "train_loss": train_loss, "train_acc": train_acc,
                "test_loss": test_loss, "test_acc": test_acc
            })
        else:
            print("epoch %d/%d, train_loss = %s, train_acc = %s" % (epoch,
                  start_epoch + num_epochs - 1, train_loss, train_acc))
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)
            hist.append({
                "train_loss": train_loss, "train_acc": train_acc
            })

        optimizer.zero_grad()
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hist": hist
        }

        # Save checkpoint.
        torch.save(checkpoint, os.path.join(
            checkpoint_path, "model_epoch%d.chkpt" % epoch)) 
        
    return train_accuracy,test_accuracy
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Block Frame&Flow ConvNet")
    parser.add_argument("--dataset_dir", type=str, default="data",
                        help="directory to dataset")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size for training (default: 64)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="number of epochs to train (default: 3)")
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="start index of epoch (default: 1)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate for training (default: 0.001)")
    parser.add_argument("--log", type=int, default=10,
                        help="log frequency (default: 10 iterations)")
    parser.add_argument("--cuda", type=int, default=0,
                        help="whether to use cuda (default: 0)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    batch_size = 64
    num_epochs = 50
    start_epoch = 1
    lr = args.lr
    log_interval = args.log

    if args.cuda == 1:
        cuda = True
    else:
       cuda = False

    print("Loading dataset")
    train_set = opticalflowdataset(dataset_dir, "train_flow")
    test_set = opticalflowdataset(dataset_dir, "test_flow")
    train_set.zero_center(train_set.mean)
    test_set.zero_center(train_set.mean)
    
    # Create model and optimizer.
    model = CNNOpticalFlow()

    if start_epoch > 1:
        resume = True
    else:
        resume = False

    # Create directory for storing checkpoints.
    os.makedirs(os.path.join(dataset_dir, "cnn_optical_flow_model_chckpts"),
                exist_ok=True)

    print("Start training")
    history,test_accuracy_val = train(model, num_epochs, train_set, test_set, lr=lr, batch_size=batch_size,
          start_epoch=start_epoch, log=log_interval, 
          checkpoint_path=os.path.join(dataset_dir, "cnn_optical_flow_model_chckpts"),
          validate=True, resume=resume, flow=True, use_cuda=cuda)
    print(history)
    print(test_accuracy_val)

    epochs = range(1,num_epochs + 1)
    plt.plot(epochs, history, 'g')
    plt.title('Training accuracy vs epoc')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.show()
    epochs = range(1,num_epochs + 1)
    plt.plot(epochs, test_accuracy_val, 'r')
    plt.title('Test accurcy vs epoch')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    print("Making optical flow features for train dataset")
    optflow_dataset(dataset="train")
    print("Making raw_dataset features for train dataset")
    raw_dataset(dataset="train")
    print("Making optical flow features for test dataset")
    optflow_dataset(dataset="test")
    print("Making raw_dataset features for test dataset")
    raw_dataset(dataset="test")
    print("Making optical flow features for validation dataset")
    optflow_dataset(dataset="validation")
    print("Making raw_dataset features for validation dataset")
    raw_dataset(dataset="validation")
    
    
    
    dataset_dir = args.dataset_dir
    model_dir = "data/cnn_optical_flow_model_chckpts/model_epoch40.chkpt"

    print("Loading validation dataset")
    train_dataset = opticalflowdataset(dataset_dir, "validation")
    video_frames = pickle.load(open("data/validation.p", "rb"))
    video_flows = pickle.load(open("data/validation_flow.p", "rb"))

    print("Loading the trained model")
    chkpt = torch.load(model_dir, map_location=lambda storage, loc: storage)
    model = CNNOpticalFlow()
    model.load_state_dict(chkpt["model"])

    # Number of correct classified videos.
    correct = 0

    model.eval()
    for i in range(len(video_frames)):
        frames = video_frames[i]["frames"]
        flow_x = [0] + video_flows[i]["flow_x"]
        flow_y = [0] + video_flows[i]["flow_y"]

        # Class probabilities.
        P = np.zeros(6, dtype=np.float32)

        current_block_frame = []
        current_block_flow_x = []
        current_block_flow_y = []
        cnt = 0

        for i_frame in range(len(frames)):
            current_block_frame.append(frames[i_frame])

            if i_frame % 15 > 0:
                current_block_flow_x.append(flow_x[i_frame])
                current_block_flow_y.append(flow_y[i_frame])

            if (i_frame + 1) % 15 == 0:
                cnt += 1

                current_block_frame = np.array(
                    current_block_frame,
                    dtype=np.float32).reshape((1, 15, 60, 80))

                current_block_flow_x = np.array(
                    current_block_flow_x,
                    dtype=np.float32).reshape((1, 14, 30, 40))

                current_block_flow_y = np.array(
                    current_block_flow_y,
                    dtype=np.float32).reshape((1, 14, 30, 40))

                current_block_frame -= train_dataset.mean["frames"]
                current_block_flow_x -= train_dataset.mean["flow_x"]
                current_block_flow_y -= train_dataset.mean["flow_y"]

                tensor_frames = torch.from_numpy(current_block_frame)
                tensor_flow_x = torch.from_numpy(current_block_flow_x)
                tensor_flow_y = torch.from_numpy(current_block_flow_y)

                instance_frames = Variable(tensor_frames.unsqueeze(0))
                instance_flow_x = Variable(tensor_flow_x.unsqueeze(0))
                instance_flow_y = Variable(tensor_flow_y.unsqueeze(0))

                score = model(instance_frames, instance_flow_x,
                              instance_flow_y).data[0].numpy()

                score -= np.max(score)
                p = np.e**score / np.sum(np.e**score)
                P += p

                current_block_frame = []
                current_block_flow_x = []
                current_block_flow_y = []

        P /= cnt
        pred = CATEGORIES[np.argmax(P)]
        if pred == video_frames[i]["category"]:
            correct += 1

        if i > 0 and i % 10 == 0:
            print("Done %d/%d videos" % (i, len(video_frames)))

    print("%d/%d correct" % (correct, len(video_frames)))
    print("Accuracy: %.9f" % (correct / len(video_frames)))

