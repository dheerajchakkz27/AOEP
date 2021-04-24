const bcrypt = require('bcrypt');
const saltRounds = Number.parseInt(process.env.SALT);
const jwt = require('jsonwebtoken');

const generateHash = async(pwd) => {
    let salt = await bcrypt.genSalt(saltRounds);
    let hash = await bcrypt.hash(pwd, salt);
    return hash;
}

const validatePassword = async(pwd, hash, salt) => {
    let isValid = await bcrypt.compare(pwd, hash);
    return isValid;
}

const issueToken = (id) => {
    let token = jwt.sign({ id, iat: Date.now() }, process.env.JWT_SECRET, { expiresIn: 600 });
    return token;
}

module.exports = {
    generateHash,
    validatePassword,
    issueToken
}