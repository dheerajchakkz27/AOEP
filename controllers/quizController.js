const Quiz = require('../models/quizModel');
const Participant = require('../models/participantModel');
const helper = require('../services/helper');

const getAllQuizzes = async(req, res, next) => {
    Quiz.find({ isEnabled: false })
        .then((quizzes) => {
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json');
            res.json(quizzes);
        })
        .catch((err) => next(err));
}

const createQuiz = async(req, res, next) => {
    Quiz.create(req.body)
        .then((quiz) => {
            console.log('Quiz Created: ', quiz);
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json');
            return res.json(quiz);
        }, (err) => next(err)).catch((err) => next(err));
}

const getQuiz = async(req, res, next) => {
    Quiz.findById(req.params.quizId)
        .then((quiz) => {
            console.log(req.params.quizId);
            console.log(quiz);
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json');
            return res.json(quiz);
        }).catch((err) => next(err));
}

const deleteQuiz = async(req, res, next) => {
    Quiz.findByIdAndRemove(req.params.quizId)
        .then((quiz) => {
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json');
            return res.json(quiz);
        }, (err) => next(err)).catch((err) => next(err));
}

const addParticipants = async(req, res, next) => {
    let participants = helper.createParticipants(req.body.emails, req.params.quizId);
    console.log(participants);
    Participant.collection.insertMany(participants)
        .then((docs) => {
            Quiz.updateOne({ _id: req.params.quizId }, { $addToSet: { participants: req.body.emails } })
                .then((doc) => {
                    console.log(doc);
                    res.json(docs);
                })
                .catch((err) => {
                    res.json(err);
                })
        })
        .catch((err) => {
            res.json(err);
        })
}

const getParticipants = async(req, res, next) => {
    Quiz.findById(req.params.quizId)
        .then((doc) => {
            res.json(doc.participants)
        })
        .catch((err) => {
            res.json(err);
        });
}

module.exports = {
    getAllQuizzes,
    createQuiz,
    getQuiz,
    deleteQuiz,
    addParticipants,
    getParticipants,
}