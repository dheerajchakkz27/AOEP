const Quiz = require('../models/quizModel');

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

module.exports = {
    getAllQuizzes,
    createQuiz,
    getQuiz,
    deleteQuiz
}