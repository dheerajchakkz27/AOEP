const mongoose = require('mongoose');
const dotenv = require('dotenv');

dotenv.config();
let uri;
if (process.env.NODE_ENV == 'development')
    uri = process.env.TEST_DB;
else
    uri = process.env.DB;

mongoose.connect(uri, {
    useUnifiedTopology: true,
    useNewUrlParser: true,
});

const connection = mongoose.connection;

module.exports = connection;