const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const connection = require('./config/databaseConfig');
const authRouter = require('./routes/authRoute');
const passport = require('./config/passportConfig');

const app = express();

require('dotenv').config();

let origin;
if (process.env.NODE_ENV === 'production')
    origin = process.env.ORIGIN_PROD;
else
    origin = process.env.ORIGIN_DEV;

app.use(cors({
    origin: origin,
}));

app.use(morgan('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(passport.initialize());

app.use('/', authRouter);

connection.once("open", () => {
    console.log("MongoDB database connection established successfully");
});

const port = process.env.PORT;
app.listen(port, () => {
    console.log(`Server running at ${port}`);
});