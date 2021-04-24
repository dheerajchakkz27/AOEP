const express = require('express');
const authController = require('../controllers/authController');
const passport = require('../config/passportConfig');
const router = express.Router();

router.post('/register', authController.registerUser);


router.post('/login', authController.loginUser);

router.get('/', (req, res) => res.send("Hello OEP"));
router.get('/protected', passport.authenticate('jwt', { session: false }), (req, res) => res.send("Hello Proteceted"))

module.exports = router;