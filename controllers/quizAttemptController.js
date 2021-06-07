const Quiz = require('../models/quizModel');
const Participant = require('../models/participantModel');
const helper = require('../services/helper');

const answer = async(req, res, next) => {
    Participant.updateOne({ _id: req.params.userId }, { $set: req.body })
        .then((doc) => {
            console.log(doc);
            res.json(doc);
        })
        .catch((err) => {
            res.json(err);
        })
}

module.exports = {
    answer
}