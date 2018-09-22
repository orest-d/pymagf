import base64

def embedded_image(path,mimetype=None,alt=None)
    if mimtype is None:
        mimetype="image/png"
    encoded=base64.b64encode(open(path,"rb").read())
    if alt is None:
        return '<img alt="%(alt)s" src="data:%(mimetype)s;base64,%(encoded)s"/>'%locals()
    else:
        return '<img alt="%(alt)s" src="data:%(mimetype)s;base64,%(encoded)s"/>'%locals()

def embedded_javascript(text):
    encoded=base64.b64encode(text)
    return '<script type="text/javascript" src="data:text/javascript;base64,%(encoded)s"></script>'%locals()

