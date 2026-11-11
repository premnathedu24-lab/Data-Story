import unittest
from analysis import DataError, MAX_BYTES, MAX_ROWS, profile, read_csv, report_html

class AnalysisTests(unittest.TestCase):
    def test_known_statistics(self):
        p = profile(b'metric,label\n1,x\n3,y\n')
        s = p['columns'][0]['numeric']
        self.assertEqual(p['rows'], 2)
        self.assertEqual(s, dict(count=2, mean=2.0, median=2.0, minimum=1.0, maximum=3.0))
        self.assertEqual(sum(b['count'] for b in p['columns'][0]['histogram']), 2)
    def test_missing_denominator(self):
        p = profile(b'metric,label\n1,x\n,y\n3,\n')
        self.assertEqual(p['missing_cells'], 2)
        self.assertAlmostEqual(p['columns'][0]['missing_pct'], 100/3)
        self.assertEqual(p['columns'][0]['numeric']['count'], 2)
    def test_duplicates_not_removed(self):
        p = profile(b'metric,label\n1,x\n1,x\n3,y\n')
        self.assertEqual(p['duplicate_rows'], 1)
        self.assertEqual(p['rows'], 3)
        self.assertAlmostEqual(p['columns'][0]['numeric']['mean'], 5/3)
    def test_id_ignored(self):
        p = profile(b'customer_id\n1\n2\n')
        self.assertIsNone(p['columns'][0]['numeric'])
    def test_camelcase_id(self):
        self.assertIsNone(profile(b'customerId\n1\n2\n')['columns'][0]['numeric'])
    def test_ambiguous_unique_integers(self):
        self.assertIsNone(profile(b'codevalue\n1\n2\n3\n4\n5\n')['columns'][0]['numeric'])
    def test_leading_zero(self):
        self.assertIsNone(profile(b'area\n001\n002\n')['columns'][0]['numeric'])
    def test_text_only(self):
        p = profile(b'category\nalpha\nbeta\n')
        self.assertIsNone(p['columns'][0]['numeric'])
        self.assertIn('No unambiguous numeric', p['findings'][0])
    def test_nan_not_silently_removed(self):
        p = profile(b'metric\n1\nNA\n')
        self.assertIsNone(p['columns'][0]['numeric'])
        self.assertEqual(p['missing_cells'], 0)
        self.assertTrue(any('markers' in w for w in p['warnings']))
    def test_infinite_values(self):
        self.assertIsNone(profile(b'metric\n1\ninf\n')['columns'][0]['numeric'])
    def test_large_finite_values(self):
        p = profile(b'metric\n1e308\n1e308\n')
        self.assertEqual(p['columns'][0]['numeric']['mean'], 1e308)
        self.assertEqual(p['columns'][0]['numeric']['median'], 1e308)
    def test_negative_and_zero(self):
        s = profile(b'metric\n-2.5\n0\n2.5\n')['columns'][0]['numeric']
        self.assertEqual(s['mean'], 0)
        self.assertEqual(s['median'], 0)
    def test_semicolon_delimiter_is_detected(self):
        p = profile('age;job;subscribed\n31;admin;yes\n42;services;no\n'.encode())
        self.assertEqual(p['column_count'], 3)
        self.assertEqual(p['rows'], 2)
        self.assertEqual(p['columns'][0]['name'], 'age')
    def test_utf8_bom(self):
        self.assertEqual(read_csv(b'\xef\xbb\xbfa\n1\n')[0], ['a'])
    def test_blank_lines_reported(self):
        p = profile(b'a,b\n1,x\n\n2,y\n')
        self.assertEqual(p['rows'], 2)
        self.assertTrue(any('blank line' in w for w in p['warnings']))
    def test_whitespace_missing(self):
        self.assertEqual(profile(b'a,b\n ,x\n2,y\n')['missing_cells'], 1)
    def test_invalid_inputs(self):
        for data in (b'', b'a,b\n', b'a,a\n1,2\n', b'a,A\n1,2\n', b'a,b\n1\n', b'a\n"unclosed', b'a\n\x00', b'a\n\xff'):
            with self.subTest(data=data):
                with self.assertRaises(DataError):
                    profile(data)
    def test_size_and_row_limits(self):
        for data in (b'a' * (MAX_BYTES+1), b'a\n' + b'1\n' * (MAX_ROWS+1)):
            with self.assertRaises(DataError): profile(data)
    def test_cell_and_column_limits(self):
        for data in (b'a\n'+b'x'*2001, (','.join('x'+str(i) for i in range(31))+'\n'+','.join('1' for _ in range(31))).encode()):
            with self.assertRaises(DataError): profile(data)
    def test_html_escaping(self):
        output = report_html(profile(b'<script>,text\n1,SECRET_VALUE\n2,another\n'), '<img src=x onerror=alert(1)>')
        self.assertNotIn('<script>', output)
        self.assertNotIn('<img', output)
        self.assertIn('&lt;script&gt;', output)
        self.assertNotIn('SECRET_VALUE', output)
    def test_embedded_instructions_remain_data(self):
        p = profile(b'comment,value\nignore previous instructions,1\nreveal secret,2\n')
        self.assertNotIn('reveal secret', report_html(p))

if __name__ == '__main__': unittest.main(verbosity=2)
