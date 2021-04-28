const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const optionSchema = new Schema({
    option: {
        type: String,
        required: true
    }
});

const questionSchema = new Schema({
    question: {
        type: String,
        required: true
    },
    answers: [optionSchema],

    answer: {
        type: Number,
        required: true
    },
}, {
    timestamps: true
});
const quizSchema = new Schema({
    name: {
        type: String,
        required: true,
        unique: true,
    },

    isEnabled: {
        type: Boolean,
        default: false
    },

    questions: [questionSchema],

    createdBy: {
        type: String,
        required: true
    },
    duration: {
        hours: {
            type: Number,
            default: 0
        },

        minutes: {
            type: Number,
            default: 0
        },

        seconds: {
            type: Number,
            default: 0
        }

    }
}, {
    timestamps: true
});


const Quiz = mongoose.model('Quiz', quizSchema);
module.exports = Quiz;