import unittest
from predict_weather_app import app  

class FlaskAppTest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_home_page(self):
        response = self.client.get('/')  
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Weather Prediction', response.data)
        self.assertIn(b'Enter a date to get the weather prediction:', response.data)

if __name__ == '__main__':
    unittest.main()  