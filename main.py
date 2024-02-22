import argparse

import loader, unify
import handclap, headbanging, sitstand


def color(message, color='red'):
    color_map = {'red':91, 'green':92, 'orange':93, 'blue':94}
    color_code = str(color_map[color])
    return f"\033[{color_code}m" + message + "\033[0m"


def main():
    parser = argparse.ArgumentParser(description='Recoginition Pipeline')
    parser.add_argument('--crop', action='store_true', help='Crop or not input video')
    parser.add_argument('--sample', type=int, default=10, help='Frame sampling for video crop')
    parser.add_argument('--save', action='store_true', help='Save or not cropped video file')
    parser.add_argument('--mode', choices=['video', 'eval'], required=True, help='Operation mode: eval, webcam, video')
    parser.add_argument('--path', type=str, help='Video Path on video mode')
    args = parser.parse_args()

    if args.mode == 'video':
        if not args.path:
            raise ValueError(color("[Error] Video Path is empty", 'red'))
        path = args.path

        if args.crop:
            print(color('[Frames Extracting with Crop]', 'blue'))
            crop_area = loader.find_crop_area(path, sampling_interval=args.sample)
            frames, fps = loader.get_crop_frames(path, *crop_area, save=args.save)
        else:
            print(color('[Frames Extracting without Crop]', 'blue'))
            frames, fps = loader.get_frames(path)

        # Hand-Clap
        print(color('\n[Hand-Clap Detecting ... ]', 'green'))
        handclap_detections = handclap.model(frames)

        # Head-Banging
        print(color('\n[Head-Banging Detecting ... ]', 'green'))
        headbanging_detections = headbanging.model(frames)

        # Sit-Stand
        print(color('\n[Sit-Stand Detecting ... ]', 'green'))
        sitstand_detections = sitstand.model(frames, fps)

        # Unifying
        print(color('\n[Unify and Make dataframe ... ]', 'red'))
        unified = unify.unify(handclap_detections, headbanging_detections, sitstand_detections)
        print(unified)
        
        unified.to_csv('./unified.csv')
        

if __name__ == '__main__':
    main()