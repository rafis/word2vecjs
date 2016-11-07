'use strict';

function Word2vec(params) {
    for(var key in params) {
        this[key] = params[key];
    }
}

Word2vec.prototype.trainModel = function() {
    return Promise.resolve().then(function() {
        
        // TODO:
    
    });
};

module.exports = Word2vec;
