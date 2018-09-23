import base64
import numpy as np
import PIL
from six.moves.urllib.request import Request, urlopen
from os.path import splitext

def embedded_image(path,mimetype=None,alt=None, id=None):
    ext = splitext(path)[1].lower()
    if mimetype is None:
        if ext == '.png':
            mimetype="image/png"
        elif ext in ('.jpg', '.jpeg'):
            mimetype="image/jpeg"

    encoded=base64.b64encode(open(path,"rb").read()).decode('ascii')
    altattr = "" if alt is None else 'alt="%s" '%alt
    idattr = "" if id is None else 'id="%s" '%id

    return '<img %(altattr)s%(idattr)s src="data:%(mimetype)s;base64,%(encoded)s"/>'%locals()

def embedded_javascript(text):
    encoded=base64.b64encode(text).decode('ascii')
    return '<script type="text/javascript" src="data:text/javascript;base64,%(encoded)s"></script>'%locals()

def fetch(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    return urlopen(req).read()

def javascript_link(url,embedded=False):
    if embedded:
        return embedded_javascript(fetch(url))
    else:
        return '<script src = "%s"></script>'%url

def image_link(path, alt=None, id=None, embedded=False):
    if embedded:
        return embedded_image(path,alt=alt,id=id)
    else:
        altattr = "" if alt is None else 'alt="%s" ' % alt
        idattr = "" if id is None else 'id="%s" ' % id
        return '<img %(altattr)s%(idattr)s src="%(path)s">'%locals()