"""Test module for dket.analytics."""

import unittest

from liteflow import vocabulary as lvoc

from dket import analytics


class TestDecode(unittest.TestCase):
    """Test case class for sentence/formula decoding."""

    def test_decode_sentence(self):
        """Decode a sentence."""
        vocabulary = lvoc.InMemoryVocabulary()
        for word in "<EOS> <UNK> the is on .".split():
            vocabulary.add(word)
        vocabulary = lvoc.UNKVocabulary(vocabulary)

        sentence = "the cat is on the mat . <EOS>".split()
        encoded = [vocabulary.index(word) for word in sentence]
        decoded_exp = "the <UNK>@1 is on the <UNK>@5 . <EOS>".split()
        decoded_act = analytics.decode_sentence(encoded, vocabulary)
        self.assertEqual(decoded_act, decoded_exp)


    def test_decode_formula(self):
        """Decode a formula."""
        shortlist = lvoc.InMemoryVocabulary()
        shortlist.add('<EOS>')
        shortlist.add('be_on')
        sentence = "the <UNK>@1 is on the <UNK>@5 . <EOS>".split()
        encoded = [
            shortlist.size() + 1,
            shortlist.index('be_on'),
            shortlist.size() + 5,
            shortlist.index('<EOS>')]
        decoded_exp = '<UNK>@1 be_on <UNK>@5 <EOS>'.split()
        decoded_act = analytics.decode_formula(encoded, shortlist, sentence)
        self.assertEqual(decoded_act, decoded_exp)


    def test_unpad(self):
        """Test the unpadding of a sequence."""
        data = [
            ([1, 2, 3, 4, 0, 0, 0], [1, 2, 3, 4, 0]),
            ([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]),
            ([1, 2, 3, 4, 5, 6, 0], [1, 2, 3, 4, 5, 6, 0])
        ]
        padding = 0
        for sequence, expected in data:
            self.assertEqual(expected, analytics.unpad(sequence, padding))


    def test_same_length(self):
        """Test the same length function."""

        data = [
            ([], [], [], []),
            ([0], [0], [0], [0]),
            ([1, 2, 3, 4, 0], [1, 0, 0], [1, 2, 3, 4, 0], [1, 0, 0, 0, 0]),
            ([1, 0, 0], [1, 2, 3, 4, 0], [1, 0, 0, 0, 0], [1, 2, 3, 4, 0]),
        ]
        for datum in data:
            first, second, exp_first, exp_second = tuple(datum)
            act_first, act_second = analytics.samelength(first, second)
            self.assertEqual(exp_first, act_first)
            self.assertEqual(exp_second, act_second)


if __name__ == '__main__':
    unittest.main()
