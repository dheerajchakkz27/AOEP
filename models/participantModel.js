const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const participantSchema = new Schema({
    email: String,
    quizId: String,
    quizAuthor: String,
    answers: Array,
    warnings: Number
});

const Participant = mongoose.model('Participant', participantSchema);
module.exports = Participant;