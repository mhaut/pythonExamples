#!/usr/bin/env python
#******************************************************************************
#  Name:     tornado.py
#  Purpose:  API Rest
#  Usage:
#    python tornado.py
# Dependence Tornado Server. sudo apt-get install python-tornado
#
#  Copyright (c) 2012
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

import tornado.ioloop
import tornado.web

class Manejador1(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world. Manejador1")

class Manejador1Loc2(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world. Manejador1 loc 1")


class Manejador2(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world. Manejador2")

class Manejador2Loc2(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world. Manejador2 loc 2")

def make_app1():
    return tornado.web.Application([
        (r"/", Manejador1),
        (r"/other", Manejador1Loc2),
    ])

def make_app2():
    return tornado.web.Application([
        (r"/", Manejador2),
        (r"/other", Manejador2Loc2),
    ])


if __name__ == "__main__":
    app1 = make_app1()
    app2 = make_app2()
    servicePort1 = 12341
    servicePort2 = 12342
    app1.listen(servicePort1)
    app2.listen(servicePort2)
    tornado.ioloop.IOLoop.current().start()
