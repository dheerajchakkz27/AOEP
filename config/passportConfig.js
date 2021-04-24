const passport = require('passport');
const User = require('../models/userModel');
const JwtStrategy = require('passport-jwt').Strategy;
const ExtractJWT = require('passport-jwt').ExtractJwt;

require('dotenv').config();
const opts = {
    jwtFromRequest: ExtractJWT.fromAuthHeaderAsBearerToken(),
    secretOrKey: process.env.JWT_SECRET
}

let verifyCallback = async(jwtPayload, done) => {
    return User.findById(jwtPayload.id)
        .then(user => {
            return done(null, user);
        })
        .catch(err => {
            return done(err);
        });
}

let jwtStrategy = new JwtStrategy(opts, verifyCallback);

passport.use(jwtStrategy);

module.exports = passport;