"""What the test bench shows: texts per level, task instructions, next steps.

Simple texts written for this test bench, graded like the ASER tool:
paragraph ≈ Grade 1, story ≈ Grade 2, five common words, five letters.
"""

from __future__ import annotations

from ..aser import Level

CONTENT = {
    "hi": {
        Level.PARAGRAPH: "मेरा नाम राजू है। मेरे घर में एक गाय है। गाय का रंग सफ़ेद है। वह हरी घास खाती है।",
        Level.STORY: (
            "रीना के पास एक छोटा कुत्ता था। उसका नाम मोती था। एक दिन मोती घर से बाहर चला गया। "
            "रीना बहुत डर गई। उसने पूरे गाँव में मोती को ढूँढा। शाम को मोती पेड़ के नीचे सोता मिला। "
            "रीना ने उसे गले से लगा लिया। अब वह मोती का पूरा ध्यान रखती है।"
        ),
        Level.WORD: "घर पानी कमल नाक बस",
        Level.LETTER: "क म स ल र",
    },
    "mr": {
        Level.PARAGRAPH: "माझे नाव सीमा आहे. माझ्या घरी एक मांजर आहे. मांजर पांढरी आहे. ती दूध पिते.",
        Level.STORY: (
            "राजूकडे एक लाल सायकल होती. तो रोज सायकलने शाळेत जायचा. एक दिवस सायकलचे चाक पंक्चर झाले. "
            "राजू खूप नाराज झाला. त्याच्या बाबांनी चाक दुरुस्त केले. राजूने बाबांना धन्यवाद दिले. "
            "आता तो सायकल नीट चालवतो."
        ),
        Level.WORD: "घर पाणी कमळ नाक बस",
        Level.LETTER: "क म स ल र",
    },
}

TASKS = {
    Level.PARAGRAPH: {
        "title": "Paragraph",
        "instruction": "Ask the child to read the whole paragraph aloud.",
        "rule": "Pass with 3 mistakes or fewer: Story next. Otherwise: Words.",
    },
    Level.STORY: {
        "title": "Story",
        "instruction": "Ask the child to read the story aloud.",
        "rule": "Pass with 3 mistakes or fewer: Story level. Otherwise: Paragraph level.",
    },
    Level.WORD: {
        "title": "Words",
        "instruction": "Ask the child to read the five words, one after another, in one recording.",
        "rule": "Pass with 4 of 5 correct: Word level. Otherwise: Letters.",
    },
    Level.LETTER: {
        "title": "Letters",
        "instruction": "Ask the child to say the five letters, one after another, in one recording.",
        "rule": "Pass with 4 of 5 correct: Letter level. Otherwise: Beginner.",
    },
}

NEXT_STEPS = {
    Level.STORY: "Reads a Grade 2 story. Move on to reading for meaning and longer texts.",
    Level.PARAGRAPH: "Reads Grade 1 text but not yet a Grade 2 story. Practise longer texts and fluency.",
    Level.WORD: "Reads words. Practise joining words into sentences.",
    Level.LETTER: "Knows letters. Practise building words: letters with matras, then short words.",
    Level.BEGINNER: "Start with letters and their sounds.",
}
