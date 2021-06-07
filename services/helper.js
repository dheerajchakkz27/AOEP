const Participant = require('../models/participantModel');

const createParticipants = (emails, quizId) => {
    let participants = [];
    console.log(quizId);
    emails.forEach((email) => {
        let participant = new Participant();
        participant.email = email;
        participant.quizId = quizId;
        participants.push(participant);
    });
    return participants;
}

module.exports = {
    createParticipants,
}