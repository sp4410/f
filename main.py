import argparse
from train import *
from trainfast import train_faster
from test import *
from resnet import ResNet

def main():
    parser = argparse.ArgumentParser(description="Command line tool for training and testing models.")

    # Add --train argument
    parser.add_argument('--train', action='store_true', help='Run the training function')

    # Add --train-faster argument
    parser.add_argument('--train-faster', action='store_true', help='Run the faster training function')

    # Add --test argument
    parser.add_argument('--test', action='store_true', help='Run the testing function')

    # Add --test-img argument and require --image-path argument
    parser.add_argument('--test-img', action='store_true', help='Run the single image testing function')
    parser.add_argument('--image-path', type=str, help='Path to the image file for single image testing')

    args = parser.parse_args()

    # Execute the corresponding function based on the arguments
    if args.train:
        train()
    elif args.train_faster:
        train_faster()
    elif args.test:
        torch.cuda.empty_cache()
        model = ResNet().to(device)
        model.load_state_dict(torch.load('./model&img/model.pt'))
        model.eval()
        test(model)
    elif args.test_img and args.image_path:
        torch.cuda.empty_cache()
        model = ResNet().to(device)
        model.load_state_dict(torch.load('./model&img/model.pt'))
        model.eval()
        test_single_img(args.image_path, model=model)
    else:
        print("No valid command provided. Use --help for more information.")


if __name__ == '__main__':
    main()
