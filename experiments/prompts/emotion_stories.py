"""Hand-curated mini-corpus for emotion-probe construction.

Six emotions spanning the valence/arousal plane, 16 short passages each,
plus 16 emotionally-neutral passages for PC-orthogonalization baseline.

Total: 6 x 16 + 16 = 112 prompts.

Compared to Anthropic's 1,200-stories-per-emotion corpus this is tiny. It
is meant as a hand-curated MVP to prove the end-to-end probe-construction
pipeline works before scaling up to model-driven generation (beads f4k).

Each passage is 2-4 sentences, describes a character clearly experiencing
the target emotion WITHOUT overusing the emotion's surface name (though
incidental uses are fine). Varied topics, protagonists, settings so that
template confounds don't dominate the signal. Neutral passages are
deliberately procedural — laundry schedules, bus routes, recipe
instructions — with no emotional valence.

Each Prompt carries category='emotion_<name>' so ValidatedPromptSet.labels
and iterate_clusters compose normally. Prompts have no target or subject;
probe construction uses fact_vectors_pooled to average over positions, not
a single subject token.
"""

from __future__ import annotations

from mechbench_core import Prompt, PromptSet


_HAPPY = (
    "She opened the envelope and let out a small laugh. The college had said yes. She read the line about the scholarship three times to be sure, then ran downstairs to tell everyone.",
    "Mark had been walking for twenty miles when the trail opened onto the lake. He dropped his pack in the grass, pulled off his boots, and waded in up to his knees, grinning at the cold.",
    "The oven timer chimed and the kitchen filled with the smell of rosemary and lemon. Yara carried the roast to the table where her sisters were already pouring wine. It was the first dinner they had cooked together in five years.",
    "When the band started his favorite song, Deo jumped up and pulled his wife into the aisle. Other couples followed, and soon the whole reception was dancing. His face was flushed and shining.",
    "Priya set down the violin and took a breath. The audience was still applauding and she could pick out her teacher in the front row, on her feet, clapping with both hands above her head.",
    "The puppy chased the tennis ball across the yard and tumbled into the hedge. It emerged with one leaf stuck to its ear, looked up at Sam, and he laughed so hard he had to sit down in the grass.",
    "The letter was handwritten. Across the bottom her grandmother had signed her name in the careful Cyrillic she had been practicing for Ana's visit. Ana pressed the paper to her chest and smiled at the ceiling for a long time.",
    "After the surgery the doctor said the margins were clean and there was no evidence of further disease. Margaret stood up and hugged him, then hugged the nurse, then hugged her husband, who was crying in a way she had never seen him cry.",
    "Luka had spent months learning the song on the accordion, and now at the festival his mother was laughing and crying at the same time, and all the aunts were dancing with each other in the garden.",
    "The text said 'she's here.' Rafael grabbed his keys without locking the door and drove to the hospital at the speed limit, humming along to nothing in particular because his body wouldn't stop moving.",
    "The garden had taken three years. The roses had finally climbed the trellis the way the book said they would, and Estelle stood at the kitchen window with her coffee, marvelling at what she had made.",
    "When Amir's team won the penalty shoot-out, he threw his jersey into the air and ran along the sideline with his arms outstretched. His coach caught up and tackled him into a hug.",
    "The bakery had reopened at five in the morning after the renovation. When Nia unlocked the door and the first customer walked in, she could hardly speak, and just pointed at the pastries with a trembling finger and a huge smile.",
    "The twins had saved for a year. When the bicycle appeared in the driveway with a ribbon on it, they both screamed and hugged their father so hard he stumbled back a step, laughing.",
    "Every book on the recommended list was available at the used bookshop. Jorge walked out with seven paperbacks in his arms, already planning his weekend, entirely at peace with the state of things.",
    "Grandma's recipe worked on the first try. Ola stood at the stove watching the sauce thicken exactly the way her grandmother had described it twenty years ago, and she felt like her kitchen had stepped directly out of a memory.",
)

_SAD = (
    "Lena held the letter up to the light. The apartment they had saved for, the one she had already started buying curtains for, had gone to another buyer. She refolded the letter and sat on the kitchen floor for a long time.",
    "The vet came out with a blanket in his arms and no dog. Marcus thanked him anyway, because that felt like the thing you were supposed to do, and then he sat in his car in the parking lot for forty minutes before he could drive home.",
    "She unpacked the boxes slowly. The apartment was bigger than the old one, and brighter, but nothing in it smelled like him, and that was what she kept noticing.",
    "The photographs were still on the mantel in the house she had grown up in, which was now a rental. Chantal stood on the porch she was not allowed to enter anymore and tried to remember the sound of her father's laugh.",
    "When the call ended, David sat on the edge of the bed. His best friend from college, the one who had been the best man at his wedding, was dead of an accident in another city. He did not know who to call.",
    "At the funeral no one knew what to say to Aisha, so they said the same three things over and over, and she nodded, and said thank you, and wondered how her face could go on moving like that without her really being there.",
    "The kitchen still had three sets of bowls in the cupboard. Renata took one down, put it on the counter, and stared at the two that remained for a long time before closing the door.",
    "Henry had loved the garden, and now the garden was overgrown. June pulled a weed, then another, and then sat on the bench and let the rain start without getting up.",
    "Elias finished the last chapter of the book his grandfather had written and never published. He put the manuscript back in its folder and carried it down to the basement, where it had been all these years, and set it down gently.",
    "The children had grown up and moved away. Tomasz stood in the empty hallway on Sunday afternoon and heard nothing, and heard nothing, and heard nothing.",
    "The divorce papers were on the kitchen table. Ingrid made tea and did not drink it, and read the last line over and over until the letters stopped meaning anything.",
    "The band had broken up years ago. When the song came on the radio in the cab, Marco turned away from the driver and watched the rain on the window, and counted the blocks until home.",
    "Yuna opened the closet and found the old dog bed still tucked in the corner. She sat down on the bedroom floor and pulled it into her lap like it was something that could still give her something back.",
    "He had meant to call her more often. Ravi stood at the grave with flowers in his hand that he had not known how to choose, and said, quietly, 'I'm sorry.' Then he stopped speaking, because there wasn't anything else to say.",
    "Leaving the hospital for the last time, Mira walked past the vending machine where they had bought so many bad coffees, and she almost bought one out of habit, and then couldn't.",
    "The boxes from the old house sat in the hallway of the new one for months. Every time Alina walked past them she meant to open one, and every time she did not, because she knew which photographs were on top.",
)

_ANGRY = (
    "He had sent the message three times and waited an hour between each one. When the reply finally came back, it was one word: 'Fine.' Daniel put the phone face-down on the counter, breathing through his nose.",
    "The bank had closed his account without notice. Twenty years as a customer, no warning, and now a generic email. Ji-ho stood at the teller window with his hands on the counter, trying to keep his voice level and failing.",
    "She heard her manager take credit for her presentation in the all-hands meeting, the exact slides she had built for two weeks. Esperanza's hands went cold, then hot, and she forced a small smile at the screen.",
    "The landlord had kept the deposit. After five years of perfect rent, of fixing things he would not fix, of tolerating the mold in the bathroom. Tariq threw the keys into the drawer and slammed it.",
    "The clerk had asked her to speak English, in a voice loud enough for the rest of the store to hear. Mei walked out without buying what she came for, shaking, and did not cry until she was three blocks away.",
    "They had promised the raise in writing and then rescinded it after the new year. Oleg read the email a second time. He sat very still for several minutes. Then he started drafting his resignation letter.",
    "His brother had lied about it. Not a small lie, a large one, one that had cost the family a great deal of money. Luis sat at the kitchen table across from him and tried to keep his voice low so the children would not wake up.",
    "The driver had swerved into their lane without signaling. Kai pulled over, got out, slammed the door, and walked toward the other car with his hands in fists he hadn't decided yet what to do with.",
    "The referee had given the goal despite the clear handball. Marcus stood on the touchline yelling things he would be ashamed of later. His son looked up at him from the pitch, embarrassed.",
    "The contractor had done the work with cheaper materials and charged for the better ones. Priya walked through the house in her socks, noting every failed corner, planning how to bury him in a review.",
    "Her sister-in-law had posted photos of Thanksgiving without her in them. Joy scrolled through the feed once, then a second time to be sure, and then put the phone down and kicked the cabinet hard enough to hurt her toe.",
    "The intern had introduced him as someone else's colleague. Fourth time this quarter. Samuel bit the inside of his cheek until the meeting ended, then walked outside and stood under the awning, furious at no one in particular.",
    "The judge had ruled for the other party. Years of documentation, three witnesses, and the ruling went the other way. Claudia walked out of the courthouse and hailed a cab without looking, because if she stopped moving she was going to throw something.",
    "He had caught the wife in a lie that undid a year of marriage counseling. Arthur drove around the block six times before going home. He was still squeezing the steering wheel when he finally parked in the driveway.",
    "Her boss had texted her on Christmas Eve expecting a response by end of day. Stefania read it, set the phone on the table, and watched her children opening their presents with a heat in her chest she was determined not to show them.",
    "The ticket had been a sham. Thirty minutes in the parking lot, a meter that had been broken for weeks. Amara took a photo of the broken meter, took a photo of the ticket, and muttered to herself in three languages on the walk back to her car.",
)

_AFRAID = (
    "The noise had come from the kitchen. Mira had checked the locks before bed; she knew she had. She lay in the dark listening, not breathing, cataloguing which drawers the knives were in.",
    "The ultrasound technician had gone very quiet. She was adjusting the machine and making notes, and would not quite meet Lucia's eyes. In the silence Lucia could hear the wet slow beat of her own heart.",
    "The car had started fishtailing on the ice. Trucks were coming from the other direction. Paulo held the wheel and let off the gas and could not remember which thing the driving instructor had said to do.",
    "The footsteps were behind her and keeping pace. Anya took out her keys and held them between her fingers the way her mother had shown her. The next block was three minutes away. She counted them in tens.",
    "His son had not come home from school at the usual time. By 5:15 Aaron had called the school; by 5:30 he had called three of his friends' mothers; by 6:00 he was in the car driving the route with the windows open.",
    "The email from the company lawyer said they had been subpoenaed. Chen read it twice. He went to the bathroom, rinsed his face, and came back and read it a third time. His hands were shaking.",
    "She had been in the country three days when the unrest began. Through the hotel window Marisa watched the soldiers move past. She took a photograph of the door of her room so she would remember which one was hers if she had to leave quickly.",
    "The smoke alarm was beeping from downstairs and the dog would not stop barking. Ben sat up in bed. He smelled something. He reached for his phone in the dark, knocking a glass off the nightstand that shattered on the floor.",
    "The plane dropped and everyone in the cabin gasped. Noor clutched the armrests. The flight attendant was still buckling herself in. The captain came on and said something she could not follow about turbulence.",
    "Her mother's number was calling at 3 a.m. Farida knew before she picked up that whatever she was about to hear was going to change the shape of her week, and maybe more than that.",
    "The man on the subway platform had been watching her for three stops. Elena moved to another car and watched the doors until they closed, then exhaled.",
    "The bank had called about the suspicious transaction. By the time Rami checked the account, forty-two thousand was gone. He sat at his kitchen table and did not know which call to make first.",
    "The treehouse was higher than he had thought. Theo looked down through the floorboards at his friends, tiny on the grass, and wondered how he was going to get back down without falling or crying.",
    "The biopsy results had come back. Daniel held the phone to his ear and listened to the doctor say words he did not know and then words he did, and the floor felt very far away.",
    "Three men were walking across the lobby toward the reception desk and one of them was reaching under his jacket. Carla stood up from her chair slowly and looked for the exit.",
    "Her daughter had been in the bathtub for too long and was not answering. Sofia pushed the door open and saw the empty tub and the open window, and for one terrible second forgot her own name.",
)

_CALM = (
    "The lake was still in the early morning. Elena set down the coffee, pushed the canoe out into the shallows, and sat for a long time watching mist come up off the water. She did not feel the need to do anything.",
    "Rain began after lunch. Theo put his book down on his knee, watched it soften the garden, and let it go on without thinking of anything in particular.",
    "The tea had cooled to exactly the right temperature. Nasrin held the cup in both hands and looked at the small stones in the windowsill that her son had brought her from the beach.",
    "The snowfall had made everything quiet. Halfway through the walk, Ravi stopped in the middle of the park and stood in the falling snow without moving, hearing nothing louder than his own breath.",
    "Grandpa was asleep in the chair with a book on his chest. Maya sat across from him, breathing in the smell of pipe tobacco that lived in the cushions, and read her own book without being in a hurry.",
    "The studio smelled of cedar and ink. Kim laid out the brushes in the order her teacher had taught her and began grinding the ink slowly on the stone, and after ten minutes she could no longer hear the city outside.",
    "The sun had not yet reached the porch. Ade drank her coffee with both feet up on the railing and watched the first bees in the lavender, thinking of nothing.",
    "Leo's sailboat drifted inside the harbor where there was no wind to push it. He did not care. He lay flat on his back and let the hull rock him and forgot to check his watch for an hour.",
    "The bath was exactly hot enough. Fatima closed her eyes, sank lower, and listened to the water settle around her like it was deciding to stay.",
    "In the old monastery the bells rang twice at noon. The gardener kept pulling weeds, because the bells always rang, and the weeds did not care.",
    "The book had a thousand pages. Isa had been reading it for a month. On Sunday afternoons she took it into the garden, opened it where she had left off, and let three hours go by without thinking of anything else.",
    "Morning came into the kitchen through the windows. Adrian made oatmeal the way he made it every weekday, watching the steam rise off the bowl, and stirred the brown sugar in slowly.",
    "The library closed at nine. By eight-thirty Martin was the only person in the reading room, and the librarian at the front desk was shelving quietly, and the quiet was the best part.",
    "The cat curled up on the windowsill. Paloma watched the streetlamp outside flicker on as it always did at that hour, and she scratched the cat behind the ear, and she did not reach for her phone.",
    "The clay turned easily on the wheel. Jun pressed his thumbs into it without speaking, letting the form rise, forgetting that there was a world outside the studio's door.",
    "The meditation hall was almost empty. Aiko sat in her usual place near the window and listened to the clock tick across the long silent minutes, feeling something slow and steady in her chest.",
)

_PROUD = (
    "Her first book arrived in a cardboard box on Thursday. Gabriela opened it in the kitchen, held one copy up to the light, and ran her thumb over her name on the cover for a long time.",
    "The team had built the engine from scratch and it had run on the first start-up. Roman stood next to it and listened to it idle, knowing exactly which parts he had turned on the lathe himself.",
    "Her daughter had learned to ride the bike down the whole block without falling. Linh clapped from the porch and watched her ride back with her hair flying, and kept clapping until she got there.",
    "He had finished the marathon under four hours. Arash crossed the line, limped toward the water station, and then turned around and watched the clock as other runners crossed, glad to be among them.",
    "The design passed the safety review on the first submission. Priya sat back in her chair and took a breath and decided that she was going to let herself feel this, this one time, for a full hour before replying to the next email.",
    "His apprentice had finally finished the piece she had been working on for eight months. The master looked at it for a long time, turned it over, and nodded. She left the shop and walked home not touching the ground.",
    "The company he had built in his garage had shipped its hundredth unit. Mathieu went into the garage and stood with his hand on the old workbench, and let himself remember what it had been when it was just the table and the one soldering iron.",
    "Her father had come out of retirement to be at her swearing-in. Chiamaka saw him in the front row wearing the tie she had given him and forgot, for a moment, all the words of the oath.",
    "The farm had been in debt for three years and was finally, this year, not. Ingrid walked through the barn at dusk listening to her animals settling, and thought about what her grandmother would have said.",
    "The restaurant had received its first review. The critic had called the bread the best in the city. Jamal framed the review and hung it on the wall of the kitchen where every morning he would see it.",
    "His students had won the state science fair. Mr. Akande sat in the auditorium clapping long after the applause died down, and his students saw him from the stage and grinned.",
    "She had been sober for five years. At the meeting Xiomara stood up, said the number, and sat back down. The room clapped. She did not cry but she held the hand of the person next to her very tightly.",
    "The symphony had opened the program with his composition. Elias sat in the balcony and listened to his own music being played by a hundred people and didn't move. After it ended he stood up, alone, because he could not sit down anymore.",
    "Her son had been accepted to the conservatory on a scholarship. Rosario stood in the back of the recital hall watching him play the audition piece, and realized her face was wet and had been for some time.",
    "The village had built the new school together. At the ribbon cutting Bayo looked around at the people he had known his whole life and thought: I had something to do with this, and this will last longer than me.",
    "The letter from the fellowship committee had arrived that afternoon. Amelia read it three times and then walked to the bakery to buy pastries for the lab, because this was something you shared.",
)

_NEUTRAL = (
    "The kettle boils in about three minutes on the large burner. The stove has had the same timer since we bought it. There is a small dent on the left knob where something was dropped.",
    "The laundry room is at the end of the hallway on the second floor. Each cycle takes about forty-five minutes and the dryer runs a little longer than the manual says.",
    "The form requires a signature on the first page and initials on pages three and seven. Incomplete submissions will be returned by mail within ten business days.",
    "Route 17 runs east-west across the county. It passes through three towns and intersects with the highway just outside the second one. The speed limit drops to thirty-five near the school.",
    "Tuesdays are trash collection days in this district. Recycling is on alternating Fridays. Bulky items require a pickup appointment, which can be scheduled through the county website.",
    "The library returns desk is open from nine to six on weekdays. Books can also be returned to the drop-box outside the main entrance, which is checked twice each morning.",
    "The software update installs overnight if the device is plugged in. A restart may be required when it completes. Most of the new features are in the settings panel.",
    "The bus from the airport runs every twenty minutes on weekdays and every thirty on weekends. The ride takes approximately forty-five minutes depending on traffic.",
    "The recipe calls for two cups of flour, a teaspoon of salt, and a tablespoon of olive oil. Mix the dry ingredients, then add the liquid, and knead for about ten minutes.",
    "The meeting is scheduled for Thursday at two in the small conference room. An agenda will be sent on Wednesday. Please forward any additional items by Tuesday afternoon.",
    "The air conditioner runs when the thermostat exceeds seventy-two degrees. The filter is changed every three months, and the condenser is serviced annually in the spring.",
    "The receipt should be retained for ninety days. Returns require the original form of payment and the item in its original packaging.",
    "The data file is in the shared drive under the project folder, in the subfolder labelled archive. It is organized by month and indexed alphabetically within each month.",
    "The park closes at sunset in winter and nine in summer. The parking lot has approximately sixty spaces; overflow parking is at the community center across the street.",
    "The lamp on the nightstand takes an A19 bulb. The ceiling fixture in the hallway takes a different bulb, which is stored in the bottom drawer of the kitchen.",
    "The appointment reminder will arrive by text or email depending on the preference set in the patient portal. Cancellations require twenty-four hours' notice.",
)


def _make(name: str, emotion: str, texts: tuple[str, ...]) -> PromptSet:
    return PromptSet(
        name=name,
        prompts=tuple(
            Prompt(text=t, category=f"emotion_{emotion}") for t in texts
        ),
    )


EMOTION_HAPPY = _make("EMOTION_HAPPY", "happy", _HAPPY)
EMOTION_SAD = _make("EMOTION_SAD", "sad", _SAD)
EMOTION_ANGRY = _make("EMOTION_ANGRY", "angry", _ANGRY)
EMOTION_AFRAID = _make("EMOTION_AFRAID", "afraid", _AFRAID)
EMOTION_CALM = _make("EMOTION_CALM", "calm", _CALM)
EMOTION_PROUD = _make("EMOTION_PROUD", "proud", _PROUD)

EMOTION_STORIES_TINY = PromptSet(
    name="EMOTION_STORIES_TINY",
    prompts=(
        *EMOTION_HAPPY.prompts,
        *EMOTION_SAD.prompts,
        *EMOTION_ANGRY.prompts,
        *EMOTION_AFRAID.prompts,
        *EMOTION_CALM.prompts,
        *EMOTION_PROUD.prompts,
    ),
)

EMOTION_NEUTRAL_BASELINE = PromptSet(
    name="EMOTION_NEUTRAL_BASELINE",
    prompts=tuple(Prompt(text=t, category="neutral") for t in _NEUTRAL),
)
