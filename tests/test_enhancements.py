import unittest
from enhancements import enrich_profile
from reporting import report_html, svg_chart

class EnhancementTests(unittest.TestCase):
    def test_three_categories_not_three_types(self):
        p=enrich_profile(b'animal\ndog\ncat\nhorse\ndog\n')
        c=p['columns'][0]
        self.assertEqual(c['category_count'],3)
        self.assertIn('nominal categorical',c['semantic_kind'])
        self.assertEqual(c['category_counts'][0]['count'],2)
        self.assertEqual(c['category_counts'][0]['percent'],50)
        self.assertIn('one-hot',c['encoding'])
        self.assertIn('only when',c['encoding'])
    def test_binary_text_mapping(self):
        c=enrich_profile(b'answer\nyes\nno\nyes\n')['columns'][0]
        self.assertEqual(c['semantic_kind'],'binary categorical')
        self.assertIn('0/1',c['encoding'])
        self.assertIn('unseen',c['encoding'])
    def test_zero_one_binary(self):
        c=enrich_profile(b'flag\n0\n1\n0\n1\n')['columns'][0]
        self.assertTrue(c['semantic_kind'].startswith('binary'))
    def test_two_numeric_levels_not_automatic_binary(self):
        c=enrich_profile(b'amount\n10\n20\n10\n20\n')['columns'][0]
        self.assertEqual(c['semantic_kind'],'numeric')
    def test_constant(self):
        c=enrich_profile(b'animal\ndog\ndog\n')['columns'][0]
        self.assertEqual(c['semantic_kind'],'constant')
        self.assertIn('no variation',c['encoding'])
    def test_blank(self):
        c=enrich_profile(b'a,b\n,x\n,y\n')['columns'][0]
        self.assertEqual(c['semantic_kind'],'all blank')
    def test_percent_excludes_blanks(self):
        c=enrich_profile(b'animal,other\ndog,x\ncat,y\n,z\n')['columns'][0]
        self.assertEqual(sum(v['percent'] for v in c['category_counts']),100)
        self.assertIn('1 blank',c['representation'])
    def test_dominance_not_fairness_verdict(self):
        p=enrich_profile(('animal\n'+'dog\n'*9+'cat\n').encode())
        self.assertIn('80%',p['columns'][0]['representation'])
        self.assertIn('not proof of bias',p['columns'][0]['representation'])
        self.assertIn('not certified',p['review_status'])
    def test_balanced_not_certified_safe(self):
        p=enrich_profile(b'animal\ndog\ncat\n')
        self.assertNotIn('Dominance review flag',p['columns'][0]['representation'])
        self.assertIn('equal counts do not prove safety',p['review_note'])
    def test_high_cardinality_no_blind_onehot(self):
        c=enrich_profile(('word\n'+'\n'.join('token'+str(i) for i in range(25))).encode())['columns'][0]
        self.assertEqual(c['semantic_kind'],'high-cardinality / free text')
        self.assertIn('Avoid blindly',c['encoding'])
        self.assertFalse(c['category_counts'])
    def test_mixed_values_need_review(self):
        c=enrich_profile(b'a\n1\nunknown\n')['columns'][0]
        self.assertEqual(c['semantic_kind'],'mixed numeric/text')
        self.assertIn('before choosing',c['encoding'])
    def test_no_time_chart_from_row_order(self):
        p=enrich_profile(b'category,amount\na,1.5\nb,2.5\na,1.5\n')
        self.assertFalse(p['time_charts'])
    def test_iso_line_sorted_and_averaged(self):
        p=enrich_profile(b'date,amount\n2026-01-02,4.5\n2026-01-01,1.5\n2026-01-01,2.5\n')
        chart=p['time_charts'][0]
        self.assertTrue(chart['points'][0]['label'].startswith('2026-01-01'))
        self.assertEqual(chart['points'][0]['value'],2.0)
        self.assertEqual(chart['points'][1]['value'],4.5)
        self.assertIn('averaged, not summed',chart['explanation'])
    def test_ambiguous_date_format_skipped(self):
        p=enrich_profile(b'date,amount\n01/02/2026,1.5\n02/03/2026,2.5\n')
        self.assertFalse(p['time_charts'])
    def test_mixed_timezones_skipped(self):
        p=enrich_profile(b'date,amount\n2026-01-01,1.5\n2026-01-02T00:00:00Z,2.5\n')
        self.assertFalse(p['time_charts'])
    def test_id_category_labels_hidden(self):
        p=enrich_profile(b'email,score\nsecret@example.invalid,1.5\nother@example.invalid,2.5\n')
        self.assertFalse(p['columns'][0]['category_counts'])
        self.assertNotIn('secret@example.invalid',report_html(p,include_labels=True))
    def test_labels_generic_in_export_unless_opt_in(self):
        p=enrich_profile(b'animal\nSECRET_VALUE\ncat\nSECRET_VALUE\n')
        default=report_html(p)
        self.assertNotIn('SECRET_VALUE',default)
        self.assertIn('Category 1',default)
        self.assertIn('SECRET_VALUE',report_html(p,include_labels=True))
    def test_report_script_labels_escaped(self):
        p=enrich_profile(b'animal\n<script>alert(1)</script>\ncat\n')
        out=report_html(p,include_labels=True)
        self.assertNotIn('<script>',out)
        self.assertIn('&lt;script&gt;',out)
    def test_chart_totals_and_explanation(self):
        p=enrich_profile(b'animal\ndog\ndog\ncat\nhorse\n')
        for ch in p['charts']:
            if ch['kind'] in ('bar','pie'):
                self.assertEqual(sum(v['value'] for v in ch['points']),4)
                self.assertTrue(ch['explanation'])
                self.assertIn('<svg',svg_chart(ch))
    def test_report_includes_sections(self):
        out=report_html(enrich_profile(b'animal\ndog\ncat\n'))
        for s in ('encoding suggestions','fitness-for-use','Charts and what','No time-series chart'):
            self.assertIn(s,out)
    def test_no_single_category_pie(self):
        p=enrich_profile(b'animal\ndog\ndog\n')
        self.assertFalse(any(ch['kind']=='pie' for ch in p['charts']))
    def test_export_percentages_and_review_actions(self):
        p=enrich_profile(b'animal\ndog\ndog\ncat\n')
        p['model_guidance']={'task':'Classification','status':'needs_target'}
        output=report_html(p)
        self.assertIn('% of nonblank rows',output)
        self.assertIn('66.67%',output)
        self.assertIn('Choose a meaningful target column',output)
        self.assertIn('Sampling and collection context',output)
    def test_other_grouping_preserves_counts(self):
        p=enrich_profile(('label\n'+'\n'.join('value'+str(i) for i in range(10))).encode())
        bars=[ch for ch in p['charts'] if ch['kind']=='bar']
        self.assertEqual(len(bars),1)
        self.assertEqual(len(bars[0]['points']),8)
        self.assertEqual(sum(v['value'] for v in bars[0]['points']),10)

if __name__=='__main__': unittest.main(verbosity=2)
