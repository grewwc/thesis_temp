from flask import Flask, request, send_from_directory, send_file, abort, render_template
import os 

def dir_listing(BASE_DIR, req_path, template_filename='files.html'):
    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = os.listdir(abs_path)
    return render_template(template_filename, files=files)