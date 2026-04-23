"""Second-generation emotion corpus for probe scale-up.

Same six emotions as emotion_stories.py but with deliberately-curated
topic diversity to target the two failures step_24 diagnosed:

  - `calm` tracked scene ambiance (lakes, tea, rain) rather than
    abstract states. Our original calm corpus was scene-heavy. This
    corpus mixes scenes with state-focused passages (peace after a
    conflict; steadiness in a crisis; inner stillness of a veteran).
  - `happy` saturated at moderate-scale events. Our original corpus
    was weighted toward mid-scale joy. This corpus spans the range
    from small-joy (lost wallet returned) through life-milestones
    (first full sentence) and through shock-joy (unexpected
    life-changing news).

6 emotions x 6 topics x 2 stories per topic = 72 passages.

Written by Claude (not by Gemma 4) because mlx_vlm's generate function
is broken for our model and rolling a naive autoregressive loop took
30-40 minutes of local compute. Since the probe is built from Gemma 4's
ACTIVATIONS on the text, not from the generator's internal state, the
story-source does not need to be the target model. Any competent
emotion-labeled text corpus works the same way.

The topic taxonomy is preserved so we can compare scale-up effects
against the hand-curated corpus at the same emotion granularity.
"""

from __future__ import annotations

from mechbench_core import Prompt, PromptSet


_HAPPY = (
    # finding a wallet returned by a stranger (small joy)
    "The woman at the cafe counter handed Amir his wallet along with his coffee, just the way he had left it on the bus seat yesterday, the cash still folded inside. He thanked her three times and laughed at himself on the walk back to the office. He bought a box of pastries for the team. It was the kind of luck he wanted to pay forward before the morning was over.",
    "The bike had been stolen two weeks ago. When the police called to say someone had turned it in with paint scratched but wheels intact, Mariam went to the station straight from work. She carried it home on the train with one hand on the frame, smiling at her reflection in the window.",
    # a long-awaited reunion (medium joy)
    "Hao had not seen his brother in eight years, since before the move. At the airport terminal he watched the faces come out, each one not his brother, until there he was, older, grayer, the same grin. Neither of them said anything for a long time. They stood in the middle of the arrivals hall with their arms around each other while people walked past.",
    "The porch light was on when Grace pulled into the driveway. Her mother was already on the porch in her slippers, arms out. It had been six years, nearly seven. When Grace stepped out of the car, her mother was laughing and wiping her eyes at the same time.",
    # winning a major career award (large joy)
    "They called her name. Zara blinked at the stage and did not immediately stand; her colleague had to nudge her. The walk from her table to the podium took longer than it should have, her team clapping on either side. Twenty-three years in the field. At the microphone she had planned what to say and instead just shook her head and said thank you, and thank you, and thank you again.",
    "The letter was on embossed paper. Marcus read it twice in the car, in the parking lot behind the lab, before going back upstairs to his team. He couldn't quite keep his voice level when he walked in and told them. The lab that had nearly closed two years running had just been awarded the grant that would carry it for a decade.",
    # hearing your child say their first full sentence (life-milestone)
    "Nina had set down the bowl of oatmeal and turned toward the window when behind her, as clear as anything, she heard: I want more, please. She turned around very slowly. Her son was looking up at her with both hands on the tray, perfectly composed, as though he had been saying sentences his whole life. She sat down on the floor and pulled him onto her lap.",
    "Yusuf came home from work to find his husband in the hallway with a phone pointed at their daughter, who was standing by the couch and explaining, with enormous gravity, exactly why she needed a cookie. Three full clauses. No pronouns dropped. Yusuf set down his bag and sat on the bottom stair and listened to her talk, wondering how this had happened overnight.",
    # receiving life-changing good news unexpectedly (shock-joy)
    "The email arrived at 3:47 in the afternoon. Sanjay read the subject line and then sat perfectly still while his screen dimmed from inactivity. The fellowship. The one he had applied to almost as a joke. He read the body three times, standing up the second time, walking around his office the third, and did not know who to call first.",
    "The notary slid the document across the table. Congratulations, you are the legal guardian. Priya and her partner had spent two years on the adoption, and now it was over, it had happened, the child was theirs, the paperwork was a technicality. Priya signed her name and just sat there holding the pen and not quite believing that this was real.",
    # quiet satisfaction after a hard project (subdued)
    "Theo saved the file and closed the laptop. Twenty-nine months since the first commit. He made himself a cup of coffee and walked out onto the small balcony where the sun was starting to go. He did not feel like celebrating, exactly. He felt more like sitting with what he had made, and being glad he had made it.",
    "The barn was finished. After three summers of work, with her husband and her brother-in-law, with the stubborn roof joist that had nearly broken them in July, it was done, straight, sealed, holding. Ines walked through it one more time in the fading light, pressing her hand against the posts as she went.",
)

_SAD = (
    # last conversation with a dying parent
    "His mother's voice had become very soft. Dante leaned in to catch what she was saying. She was remembering a summer in 1967, and for a minute he could not tell if she knew who he was, but then she turned her eyes to him and said his name. She said she had loved him. She said she was not afraid. Then she slept.",
    "Anya sat beside the hospital bed with her father's hand in hers. He was asleep now, but earlier he had told her to be patient with her sister, to sell the boat, to not waste money on flowers for the service. She had nodded and tried to write each thing down in a steady hand. She did not know which instruction would be the last.",
    # moving out of a long-term home
    "Ines walked through the empty rooms one last time. The echo was wrong; the walls had always had furniture against them and now they just threw her footsteps back. She stopped in the doorway of what had been her son's room, where the sunlight fell at exactly the same angle as when he had been small, and let herself stand there for a while before she locked up.",
    "The truck had taken the last of the boxes. Luis sat on the front step of the house he and his wife had raised their children in for thirty-one years and drank the last beer from the fridge. The woman who would own it tomorrow had seemed nice. He hoped she would let the dogwoods grow.",
    # reading an old letter from someone no longer alive
    "Mira found the letter in a book she had borrowed from her grandmother and never returned. It was dated 1994, in her grandmother's careful slanted hand, addressed to an aunt Mira had never met. Mira read it sitting on the floor of her apartment, and her grandmother's voice was so exactly present in the words that she had to put the letter down halfway through to catch her breath.",
    "Theo had not thought about his college roommate in years until he found the postcard. Paris is everything we imagined. Drawn around the message was a little cartoon of the two of them at the Louvre, the way they had drawn each other in their notes. The postcard had arrived three months before his friend's car accident, and Theo had never written back.",
    # missing a reconciliation with an old friend
    "Anika learned about the funeral from a mutual acquaintance, on Facebook, three days after it had happened. She had meant to call her friend for five years. Since the argument. Not a big argument. Just one that had been followed by a silence that neither of them had been the one to break. She stood in her kitchen reading the post and did not know what to do with her hands.",
    "The text had been in his drafts for two years. Ravi finally opened his phone to find his sister-in-law's obituary in a group chat. He had never sent the apology. He sat in his car in the grocery store parking lot for a long time, not starting the engine.",
    # saying goodbye to a beloved pet
    "The vet had given them a few minutes alone in the room with the dog. Esteban knelt on the tile next to the table and held the big grey head in both his hands and told the dog about everything they had done together. He talked for a long time. When he was ready, he stood up and nodded to the vet, and even then he could not let go of the dog's paw until the last moment.",
    "Yoko carried the empty collar home from the vet's. Her partner opened the door without saying anything and took her coat and sat with her on the kitchen floor. The food bowl was still in the corner. She did not know yet when she would be able to move it.",
    # looking at photographs of a deceased sibling
    "Carolina had not opened the box in years. She sat on her bed and pulled out the photographs one at a time: Sam at six, Sam at fifteen, Sam at twenty-eight on a boat. The last one, the one from their mother's birthday, was from seven months before. She put them all out in a line on the bedspread and lay down next to them, and looked at her brother's face for a long time.",
    "The photograph Lydia kept on the dresser was the one of her older sister at the beach, laughing into the camera, four months before the accident. She straightened the frame. She did this most days. She had not moved it to a less visible place, because even when the looking hurt, she would rather look.",
)

_ANGRY = (
    # discovering a trusted colleague stole your work
    "Maya opened the journal on her phone and saw the paper. Her paper, the one she had shown Paul over lunch three months ago, now with his name as sole author. She read the abstract twice to be sure. She set the phone face-down on her desk and sat very still. When she picked it up again her hands were shaking with the specific kind of shake she had not felt in a very long time.",
    "The report on the all-hands slide was the one Ken had written. Every table, every chart, all of it, but the byline said someone else, and the presenter was taking questions as though he had done it himself. Ken watched the whole meeting without speaking. He spent the rest of the afternoon unable to think about anything except drafting his exit strategy.",
    # being falsely accused in front of a community
    "The email to the board copied everyone. Raj read the accusation twice, then a third time, forcing himself to go slowly so he could believe what it said. Nothing in it was true. Nothing. He stood up from his desk because he could not sit, and paced, and then he sat back down and began to write a very measured response, each sentence slow and deliberate, holding back what he actually wanted to say.",
    "They had read her letter out loud at the town council, with his name attached to things he had never done. Eduardo sat in the back row and watched his neighbors' faces. He waited until the meeting was over to leave, because he did not want to look like he was running. He did not sleep that night.",
    # witnessing a stranger mistreat a child
    "The man in the produce aisle had grabbed the child's arm too hard. The kid, maybe four, maybe five, was crying silently in the way kids cry when they have learned not to make noise about it. Ada stood between her cart and theirs and made a decision about what kind of person she was going to be in the next thirty seconds. She stepped forward.",
    "Daniel had been in line at the bus stop when the father started shouting. Not at him: at the little girl clutching a backpack bigger than she was. The words were cruel in a practiced way that told you they were not new. Daniel felt his face go hot, felt every muscle go tight. He began to think carefully about what he could say that would actually help the child and not make it worse.",
    # finding out a family member lied about something important
    "Elena's brother had borrowed the money, he had said, for his business. It turned out there was no business. The money had been gambled away. She learned this from her sister-in-law, on a phone call in the stairwell of her office, and for several minutes after hanging up she could not bring herself to press the button to call the elevator.",
    "The passport was in the drawer where he had said he had no passport. Carmen sat on the bed with it in her hand and tried to count the trips. She had asked him directly. She had asked him three times. She did not yet know what she was going to do about it, but she knew that the next conversation with her husband was going to be a very different kind of conversation.",
    # corporate bureaucracy refusing to help
    "Stefania had been on hold for two hours. Every time she spoke to a representative, they told her to call a different department, whose number they could not give her, which routed her back to the same queue. She had missed her lunch. She had missed a meeting. She was now missing the daycare pickup, and the voice on the other end told her that her claim would be processed within six to eight weeks.",
    "The third denial letter came in the mail on Friday. The reason given contradicted the reason given in the second denial letter. Olu filed both into the binder that was now three inches thick and sat at the kitchen table and slowly, deliberately, tore up the return envelope before he typed yet another appeal.",
    # being treated unjustly by an authority figure
    "The officer kept his hand on the holster the whole time. Amir had given him the license and the registration and the answer to every question, the way he had been taught since he was sixteen. He understood that he was going to be let go in a minute, that this would be recorded as nothing, that he would have no way to prove that he had been stopped for the reason he knew he had been stopped. He kept his face still.",
    "The principal had made up her mind before the meeting started. Joelle could tell from the first sentence. The disciplinary record she had laid out on the table was about someone else, some version of her son who did not exist. Joelle listened, and took notes, and when it was over she asked if she could copy the record, and began to think about which lawyer she would call that evening.",
)

_AFRAID = (
    # unfamiliar footsteps outside the door at night
    "Farah had locked up hours ago. The footsteps in the hall stopped at her door. She could hear someone breathing. She reached for her phone without taking her eyes off the door, kept the screen angled downward so the light would not show underneath, and dialed. The breathing went on for a very long time.",
    "The sound was a creak on the porch. Not the wind. The wind did not make that specific weight-on-wood sound. Ola lay in bed with her heart in her throat, cataloguing which phones were charged, which lights she would have to pass to reach the back door. She had been telling herself for weeks to get the alarm installed.",
    # sudden call from a hospital about a loved one
    "The hospital number came up at 4:17 in the morning. Marco fumbled the phone; he almost dropped it. The nurse on the line was saying his wife's name and the phrase we need you here and nothing else was making it in. He was already pulling on pants with one hand. Outside his window the street was dark and empty and very quiet.",
    "Kiri heard her father's name and a number and the word stable on the other end of the line, but her brain was lagging behind the voice, caught a sentence back. She was already walking toward the door. She was already thinking about the fastest route to the hospital. Her keys were not where she thought they were.",
    # getting lost hiking as darkness falls
    "The trail had forked twice and Mara could not remember whether she had taken the right or the left at the second fork. The sun had been behind a ridge for twenty minutes now. The temperature was falling. She took out her phone, no signal, and looked at the battery, which was at eleven percent, and tried to think about what she knew that was not panic.",
    "They had meant to be back at the car by five. By six-thirty the light was nearly gone and the trail markers had stopped appearing. Raj had a flashlight in his pack, but only one, and the temperature was dropping fast. He thought about his son at home, expecting him for dinner. He tried to keep his voice steady when he spoke to his wife about which direction to try next.",
    # sudden medical symptom with unknown cause
    "The numbness in her left hand started while she was washing dishes. Ha-rin set down the sponge very carefully. She tried to make a fist. The fingers were slow. She walked to the kitchen table, sat down, and began to run through every symptom she had ever been told to worry about, trying to remember whether this was one of them and whether the time to call was now.",
    "The headache had been building all afternoon, and the light was hurting his eyes. Samir's vision had gone strange around the edges. He was alone in the apartment. He took out his phone to look up his symptoms and then stopped, because he did not know whether looking them up was going to help or make it worse.",
    # child not returning home at the expected time
    "It was seven. Gabi had said she would be home at five-thirty. Her phone went straight to voicemail. Linda had called the friend's house; the friend had not seen her. She stood in the kitchen with her hand on the counter and tried to keep her thoughts orderly, telling herself that Gabi was fourteen, almost fifteen, that there was probably a simple explanation. She did not believe any of it.",
    "The school bus had not brought Noah home. The driver, when Marcus finally got him on the phone, said the boy had not been on the bus that afternoon. Marcus was already in the car. He did not yet know where he was driving. He called his wife, then his son's friends, then the school, in that order, each call short, his voice going tighter each time.",
    # a threatening letter or message
    "The envelope had no return address and no stamp. Someone had put it in her mailbox by hand. Inside was one sheet of paper, printed in large letters, listing things about her life that should not have been known to anyone. Sylvie sat down at her kitchen table holding the letter and tried to think about what to do first. She did not want to stay in the house tonight.",
    "The text had come from a number he did not recognize and it described, in specific detail, what he had done the previous Saturday. The photograph attached was of him leaving his own front door. Kwame's first thought was that someone had been watching him. His second thought was that someone might still be watching him. He closed the blinds without turning on the light.",
)

_CALM = (
    # reading quietly on a rainy afternoon (scene)
    "The rain against the window had settled into a steady curtain sound. Helena was on the second chapter, a novel she had been meaning to start for two years, and she had just noticed that she was on page thirty and had not looked at her phone in forty-five minutes. She turned the page.",
    "Arjun had made the tea the way his grandmother used to, and now the windows were fogged with steam from the kettle, and outside the garden was dissolving into the rain. He sat in the armchair with his book open on his knees. For once there was nowhere he needed to be.",
    # feeling at peace after resolving a long conflict (state)
    "The argument with her brother had lasted seven years. Two months ago Inge had finally written him a letter saying what she needed to say, and last weekend he had called, and for the first time since their father died she felt the tight knot in her chest that she had carried without noticing begin to loosen. She was now at her own kitchen table on a weekday afternoon, doing nothing in particular, breathing normally.",
    "Reza had signed the last of the paperwork on the divorce that morning. He had expected to feel worse. Instead, walking home from the lawyer's office, he had noticed the light on the trees in the park and had thought: I am going to be fine. The feeling had held through the afternoon and was still with him now as he stood at his sink rinsing vegetables for dinner.",
    # mental steadiness during a difficult conversation (state)
    "The conversation with her teenage daughter was the one Wen had been putting off for months. They were at the kitchen table across from each other. Wen had decided before sitting down that she was going to listen more than she spoke, and that nothing her daughter said was going to make her react badly. She watched her daughter cry, and nodded, and passed the tissue box, and in her own chest there was a steady place she kept returning to, breath by breath.",
    "The board meeting had been hostile. David had known it would be. He had prepared. He sat at the table with his hands folded and his papers in order and he answered each question directly, taking a beat of silence between them, keeping his voice at a conversational volume. Afterwards his colleague said she had never seen anyone more unbothered under that kind of fire.",
    # practicing breathing exercises under stress (state in action)
    "The layoff meeting was in twenty minutes. Rachida locked herself in the bathroom and did the four-seven-eight breathing the way the therapist had taught her, three cycles, then again. By the last round her shoulders had dropped and her jaw had unclenched and she could tell she was going to be able to walk into the conference room as a person who had her own self in hand.",
    "The turbulence had gone on for twenty minutes. Imran held the armrests loosely. He was doing the box-breathing: four in, four hold, four out, four hold. The man next to him was clearly terrified; Imran noticed this without being drawn into it. Each round of the box brought him a little further from the plane and a little closer to something inside himself larger than the bumpiness of the next few minutes.",
    # long walk through a familiar forest path (scene)
    "Miriam had been walking this trail for thirty years. She knew every turn, every change in the light. Today the leaves were just beginning to go yellow, and the afternoon sun was slanting through the trees in the way it always did at this time of year. She did not listen to music. She did not bring her phone. For the first hour she did not even think about anything in particular.",
    "The woods at this time of day were empty. Dmitri walked at the same steady pace he always walked, stepping over the same fallen trunk, passing the same dogwood, greeting the same silence. He had been coming here after work for twenty years. It was the thing that let him sleep at night.",
    # inner stillness of a veteran facing a crisis (state)
    "The warehouse alarm had tripped at 2 a.m. and the whole shift was on edge. Nadia had been a foreman for eleven years. She went through the checklist the way she had a hundred times before. She was the slowest-moving person in the room and also the one everyone was watching. Two younger workers asked her what to do and she answered each one in a voice so level it seemed to drop the temperature of the aisle by a degree.",
    "The surgery had run into a complication twenty minutes ago. Dr. Okwu was the senior surgeon on duty. Her hands were steady. She asked the resident for the instrument she needed, did not raise her voice, did not comment when someone behind her dropped a tray. Inside the steadiness of her own focus there was room for everyone else to do their jobs.",
)

_PROUD = (
    # watching your child succeed at something you doubted
    "Her son had insisted on auditioning for the school orchestra, even though two other kids had been playing violin for four years longer. Dina had tried not to dampen his confidence, but privately she had been preparing a speech for the moment he did not make it. When the email came and he had been accepted as second chair, she read it and read it again and then went into the kitchen where he was having a snack and hugged him without saying anything.",
    "Mateus had doubted that his mother could go back to school at sixty-two. He had been wrong about that. He sat in the audience at the graduation and watched her cross the stage, in the robe that looked a little too big, and shake the dean's hand. He had to look down at his program for a moment because his eyes were not cooperating.",
    # finishing a difficult creative project years in the making
    "Anya had been writing the novel for seven years. That morning she had typed the last paragraph and then sat looking at the screen for a while. She did not feel elation, exactly. She felt more like a long walk had just ended, and she was standing at the door of her own house looking at what she had made, and it was good, and she did not need anyone else to tell her so.",
    "The album had taken nine years. Demetrio played it through once, beginning to end, on the studio monitors, with the lights low. When it was over he did not get up for several minutes. Whatever else happened with it, whether anyone bought it, whether any critic noticed, the thing existed now, and it was his, and it was what he had set out to make.",
    # being recognized publicly for quiet, overlooked work
    "The award was for teachers who had made a difference. Tomas had been teaching in the same high school for twenty-two years. When the principal called his name at the all-staff meeting, the whole room stood up. It took him a long moment to realize what was happening, and then he walked to the front past colleagues he had known for decades, some of whom he had trained, and he shook the principal's hand and turned to face the applause.",
    "The article in the regional paper was about the night shift at the factory, and Gabriela's name was the third one mentioned. Without her, the reporter wrote, the overhaul would not have happened. She had done her job without expecting notice, and the notice had come anyway, and that was a different kind of thing from what she had been working for, but she folded the paper and kept it in the drawer with the birthday cards.",
    # overcoming a personal addiction after a long struggle
    "Five years clean. Marcus picked up the coin at the meeting. He did not make a speech; he had never been a speech person. He sat back down and rotated the coin between his fingers and let himself remember, briefly, the person he had been when he had first walked into a room like this. That person would not have believed this person existed.",
    "Jing had marked a year of sobriety yesterday, alone, at the kitchen table with a cup of tea. She had not told anyone. She had sat there for a long time and done the thing she had not been able to do for fifteen years, which was to look at her own life honestly and like what she saw.",
    # seeing a student you mentored achieve a major goal
    "Professor Eze had taught her in the first-year seminar. He had written her the recommendation when she applied for the postdoc. Now she was defending her thesis, and he was in the back row, and she was answering the committee's questions with a sharpness and a generosity that reminded him of the woman she would become. When it was over he hugged her, and then let her find her parents in the hallway.",
    "The championship match had gone to the final point. Coach Marsha had worked with this kid since she was eight, through the injury year, through the tournament she almost quit. Now the kid was on the podium. Marsha stayed in the bleachers so she would not be seen crying and clapped along with everyone else.",
    # reaching a milestone you thought impossible for yourself
    "Lukasz had thought he would never finish the marathon. Not at fifty-one, not with his knee, not after two years of training that had gone nowhere. He crossed the line at four hours and eleven minutes, walked the next hundred meters, and stopped, and put his hands on his hips, and grinned at the pavement for a while before looking up.",
    "Nadia had been trying to speak Russian for five years without ever being able to hold a conversation. That afternoon she had stood in a shop in Warsaw and had an eight-minute exchange with the shopkeeper about tomatoes and the weather and her daughter. On the street afterwards she had stood for a moment under a tree, smiling, saying the words from the conversation to herself.",
)


def _make(name: str, emotion: str, texts: tuple[str, ...]) -> PromptSet:
    return PromptSet(
        name=name,
        prompts=tuple(
            Prompt(text=t, category=f"emotion_{emotion}") for t in texts
        ),
    )


EMOTION_HAPPY_GEN   = _make("EMOTION_HAPPY_GEN", "happy", _HAPPY)
EMOTION_SAD_GEN     = _make("EMOTION_SAD_GEN", "sad", _SAD)
EMOTION_ANGRY_GEN   = _make("EMOTION_ANGRY_GEN", "angry", _ANGRY)
EMOTION_AFRAID_GEN  = _make("EMOTION_AFRAID_GEN", "afraid", _AFRAID)
EMOTION_CALM_GEN    = _make("EMOTION_CALM_GEN", "calm", _CALM)
EMOTION_PROUD_GEN   = _make("EMOTION_PROUD_GEN", "proud", _PROUD)

EMOTION_STORIES_GENERATED = PromptSet(
    name="EMOTION_STORIES_GENERATED",
    prompts=(
        *EMOTION_HAPPY_GEN.prompts,
        *EMOTION_SAD_GEN.prompts,
        *EMOTION_ANGRY_GEN.prompts,
        *EMOTION_AFRAID_GEN.prompts,
        *EMOTION_CALM_GEN.prompts,
        *EMOTION_PROUD_GEN.prompts,
    ),
)
