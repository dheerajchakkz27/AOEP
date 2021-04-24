const authService = require('../services/authService');
const User = require('../models/userModel');

const registerUser = async(req, res, next) => {
    let email = req.body.email;
    let password = req.body.password;
    let fullname = req.body.fullname;
    if (email === undefined || password === undefined) {
        return res.status(422).json({
            error: true,
            msg: 'Invalid data'
        });
    }
    let hashpwd = await authService.generateHash(password);
    User.findOne({ email: email })
        .then((user) => {
            if (user == null) {
                User.create({
                    email: email,
                    hashpwd: hashpwd,
                    fullname: fullname
                }).then((user) => {
                    return res.status(200).json({
                        registration: 'success',
                        user: user
                    })
                }).catch((err) => next(err));
            } else {
                return res.status(409).json({
                    error: true,
                    msg: 'User already exists'
                });
            }
        }).catch((err) => console.log(err));

}

const loginUser = (req, res, next) => {
    let email = req.body.email;
    let password = req.body.password;
    if (email === undefined || password === undefined) {
        return res.status(422).json({
            error: true,
            msg: 'Invalid data'
        });
    }
    User.findOne({ email: email })
        .then(async(user) => {
            if (!user) {
                return res.status(401).json({
                    error: true,
                    msg: 'Invalid Username or Password'
                });
            }
            let isValid = await authService.validatePassword(password, user.hashpwd, user.salt);
            if (isValid) {
                let jwt = authService.issueToken(user._id);
                return res.status(200).json({
                    error: false,
                    msg: 'Login Successful',
                    token: jwt,
                    user_id: user._id,
                    user: user
                });
            } else {
                return res.status(401).json({
                    error: true,
                    msg: 'Invalid Username or Password'
                });
            }
        })
        .catch((err) => {
            console.log(err);
            return res.status(500).json({
                error: true,
                msg: 'An unexpected error occured. Please try again later'
            });
        });
}

module.exports = {
    registerUser,
    loginUser,
}