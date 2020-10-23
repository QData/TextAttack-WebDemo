import re

class HtmlHelper():
    INITIAL_INSTRUCTIONS_HTML = """
                                <p style="font-size:1.em; font-weight: 300">
                                    👋 Welcome to the TextAttack demo app! Please select a model and an attack recipe from the dropdown.
                                </p> 
                                <hr style="margin: 1.em 0;">
                            """

    improvements = {
        "color = bold": 'style="font-weight: bold;"',
        "color = underline": 'style="text-decoration: underline;"',
        '<font style="font-weight: bold;"': '<span style=""', # no bolding for now
        '<font style="text-decoration: underline;"': '<span style="text-decoration: underline;"',
        "</font>": "</span>",

        # colors
        ': red': ': rgba(255, 0, 0, .7)',
        ': green': ': rgb(0, 255, 0, .7)',
        ': blue': ': rgb(0, 0, 255, .7)',
        ': gray': ': rgb(220, 220, 220, .7)',

        # icons
        "-->": "<span>&#8594;</span>",
        "1 (": "<span>&#128522;</span> (",
        "0 (": "<span>&#128543;</span> (",
    }
    
    def __init__(self):
        pass

    @staticmethod
    def improve_result_html(result_html):
        result_html = re.sub(r"<font\scolor\s=\s(\w.*?)>", r'<span style="color: \1; padding: 1.2px; font-weight: 600;">', result_html)
        for original, replacement in HtmlHelper.improvements.items():
            result_html = result_html.replace(original, replacement)
        return result_html

    @staticmethod
    def get_attack_result_status(attack_result):
        status_html = attack_result.goal_function_result_str(color_method='html')
        return HtmlHelper.improve_result_html(status_html)

    @staticmethod
    def get_attack_result_html(idx, attack_result):
        result_status = HtmlHelper.get_attack_result_status(attack_result)
        result_html_lines = attack_result.str_lines(color_method='html')
        result_html_lines = [HtmlHelper.improve_result_html(line) for line in result_html_lines]
        rows = [
            ['', result_status],
            ['Input', result_html_lines[1]]
        ]
        
        if len(result_html_lines) > 2:
            rows.append(['Output', result_html_lines[2]])
        
        twitter_share = "<a href='https://twitter.com/intent/tweet?text=" + HtmlHelper.build_tweet(attack_result) + "' target='_blank'>" + HtmlHelper.twitter_svg() + "</a>"
        table_html = '\n'.join((f'<b>{header}:</b> {body} <br>' if header else f'{body} <br>') for header, body in rows)
        div_html = [
            f'<div style="background:white; border-radius:8px; padding:1em; padding-left:2em; box-shadow:2px 2px 9px #0000001f; margin-bottom:1em; position:relative;">', 
                f'<h3>Result {idx+1}</h3>',
                f'{table_html} <br>',
                f'<div style="position:absolute; top:1em; right:2em">{twitter_share}</div>',
                f'<div style="position:absolute; bottom:1em; right:2em; color:#c5c5c5">Thursday, October 22, 04:37 AM</div>',
            f'<div>'
        ]
        return ''.join(div_html)

    @staticmethod
    def twitter_svg():
        return '<svg xmlns="http://www.w3.org/2000/svg" fill="gray" width="24" height="24" viewBox="0 0 24 24"><path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z"/></svg>'

    @staticmethod
    def build_tweet(attack_result):
        res = "Check out the TextAttack Webapp here: https://textattack.com\n\n" + str(attack_result)
        return str(res).strip().replace('%', '%25').replace(' ', '%20').replace('\n', '%0A')
