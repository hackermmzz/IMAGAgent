import argparse
import os
#############################################
def main():
    # 1. Create argparse parser with clear description
    parser = argparse.ArgumentParser(
        description="IMAGAgent CLI Script - Invoke ProcessImageEdit for image processing",
        epilog="Example Usage: python run.py ./test.jpg 'change the background to a forest' ./output"
    )

    # 2. Add positional arguments (match your function's 3 parameters exactly)
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path of the image file to process (absolute/relative, e.g., ./photo.png or C:/images/test.jpg)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image editing (e.g., 'remove the cat and change dog's color to blue', 'move the man to right while transforming background to a beach')"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./output",
        help="Directory for saving debug logs and edited results (auto-created if missing)"
    )

    # 3. Parse command line arguments
    args = parser.parse_args()
    # 4. Validate image path (critical pre-check)
    if not os.path.exists(args.img_path):
        parser.error(f"Image path does not exist: {args.img_path}")
    if not os.path.isfile(args.img_path):
        parser.error(f"Not a valid file path (is a directory): {args.img_path}")
    # 5. check the directory exists, if not create it
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    os.system(f"rm -rf {args.dir}/*")
    # 5. Call the core processing function
    from main import ProcessImageEdit
    ProcessImageEdit(args.img_path, args.prompt, args.dir)
    # 6. Indicate completion
    print("Image processing completed. Check the output directory for results.")
#############################################
if __name__ == "__main__":
    main()