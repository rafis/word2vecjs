'use strict';

var fs = require('fs');

function Fd(fd) {
    if ('string' == typeof fd) {
        fd = fs.openSync(fd, 'r');
    }
    this.fd = fd;
    this.position = 0;
}

Fd.prototype.getc = function() {
    if (this.ch) {
        var ungetc = this.ch;
        this.ch = null;
        return ungetc;
    }
    
    var ch = new Buffer(1);
    var bytesRead = fs.readSync(this.fd, ch, 0, 1, this.position);
    if (0 == bytesRead) {
        return null;
    }
    this.position += bytesRead;
    return ch.toString('utf8');
};

Fd.prototype.ungetc = function(ch) {
    this.ch = ch;
};

Fd.prototype.close = function() {
    fs.closeSync(this.fd);
};

module.exports = Fd;
