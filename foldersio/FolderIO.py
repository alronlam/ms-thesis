from pathlib import Path

def get_files(dir_path, is_recursive, extension):
    p = Path(dir_path)

    # print("Recursed: {}, {}, {}".format(dir_path, is_recursive, extension))

    # All files in the folder
    files = [file for file in list(p.iterdir())
              if file.is_file() and (extension is not None and file.name.endswith(extension))]

    # Base Case: Return files if not recursive
    if not is_recursive:
        return files
    # Recursive Case: Keep on listing files within sub-directories and append them to the files at this level
    else:
        dirs = (dir for dir in list(p.iterdir())
                if not dir.is_file())

        for dir in dirs:
            # print (dir.name)
            files = files + get_files(dir, is_recursive, extension)

        return files





