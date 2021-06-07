const express = require('express');
const quizController = require('../controllers/quizController');
const quizAttemptController = require('../controllers/quizAttemptController');
const passport = require('../config/passportConfig');
const router = express.Router();

router.route('/')
    .get(quizController.getAllQuizzes)
    .post(passport.authenticate('jwt', { session: false }), quizController.createQuiz)
    .put((req, res, next) => {
        res.statusCode = 403 /*Not supported*/
        res.end('PUT operation not supported on /quizes');
    });
// router.delete(authenticate.verifyUser, (req, res, next) => {
//     Quizes.remove({})
//         .then((resp) => {
//             res.statusCode = 200;
//             res.setHeader('Content-Type', 'application/json');
//             res.json(resp);
//         }, (err) => next(err)).catch((err) => next(err));
// });

router.route('/:quizId')
    .get(quizController.getQuiz)
    .post((req, res, next) => {
        res.statusCode = 403 /*Not supported*/
        res.end('POST operation not supported on /quizes/' +
            req.params.quizId);
    })
    .put((req, res, next) => {
        res.statusCode = 403 /*Not supported*/
        res.end('PUT operation not supported on /quizes/' +
            req.params.quizId);
    })
    .delete(passport.authenticate('jwt', { session: false }), quizController.deleteQuiz);


router.route('/:quizId/participants')
    .get(quizController.getParticipants);

router.route('/:quizId/addParticipants')
    .post(quizController.addParticipants);

router.route('/:quizId/attempt/:userId')
    .put(quizAttemptController.answer);

module.exports = router;