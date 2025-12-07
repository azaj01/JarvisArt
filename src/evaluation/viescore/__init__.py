import sys
sys.path.insert(0, 'viescore')

from viescore_utils import (
    mllm_output_to_dict
)
import math

class VIEScore:
    def __init__(self, backbone="gpt4o", task="t2i", key_id=None, vllm_ture=True) -> None:
        self.task = task
        self.backbone_name = backbone

        if self.task not in ["t2i", "tie", "t2v"]:
            raise ValueError("task must be either 't2i' or 'tie'")

        if self.backbone_name == "qwen25vl":
            from mllm_tools.qwen25vl_eval import Qwen25VL
            self.model = Qwen25VL(key_id, vllm_ture)
        elif self.backbone_name == "google":
            from mllm_tools.google_eval import GoogleEval
            self.model = GoogleEval(vllm_ture)
        else:
            raise NotImplementedError("backbone not supported")
        self.SC_prompt = """ 
        You are a post-production specialist with expertise in enhancing photographic imagery through advanced digital editing techniques. We now need your help to evaluate the performance of an AI-powered image post-editing tool for photography.
        
        INPUTS:
            1. Two images will be provided: The first being the original photographic image and the second being an edited version of the first.
            2. The editing instruction will be provided: The post-editing needs of photographic images expressed by users with no image processing knowledge.

        METRICS (From scale 0 to 10): 
            User Instruction Satisfaction Score: A score from 0 to 10 will be given based on how well the edits follow the user's instructions.
                - Users typically have both global and local editing requirements when working with photographic images. Therefore, this score should be evaluated holistically, taking into account the user's needs for both local and global adjustments, with equal importance given to each.
                - 0 indicates that the edited image does not follow the editing instruction at all. 
                - 10 indicates that the edited image follow the editing instruction text perfectly.
            Content Consistency Score: A second score from 0 to 10 will rate the consistency of image content before and after editing. 
                - Need to compare before and after images to assess content consistency.
                - The edited image should maintain consistency in key visual elements such as the shape of landscapes, human figures (including posture, gender, and appearance), building structures, and other important features.
                - The edited image needs to maintain the consistency of local details, such as letters on clothes, textures of buildings, etc.
                - 0 indicates that the content of the image before and after editing is completely inconsistent. 
                - 10 indicates that the content of the edited image is exactly the same as the original image.
            

        You will have to give your output in this way (Keep your reasoning concise and short.):
        {
        "score" : [...],
        "reasoning" : "..."
        }
        Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the User Instruction Satisfaction Score and 'score2' evaluates the Content Consistency Score.
        Editing instruction: <instruction>
            """
        
        self.SC_prompt_local = """ 
        You are a post-production specialist with expertise in enhancing photographic imagery through advanced digital editing techniques. We now need your help to evaluate the performance of an AI-powered image post-editing tool for photography.
        
        INPUTS:
            1. Two images will be provided: The first being the original photographic image and the second being an edited version of the first.
            2. The editing instruction will be provided: The post-editing needs of photographic images expressed by users with no image processing knowledge.
        
        METRICS (From scale 0 to 10): 
            User Instruction Satisfaction Score: A score from 0 to 10 will be given based on how well the edits follow the user's instructions.
                - Users typically have both global and local editing requirements when working with photographic images. Therefore, this score should be evaluated holistically, taking into account the user's needs for both local and global adjustments, with equal importance given to each.
                - 0 indicates that the edited image does not follow the editing instruction at all. 
                - 10 indicates that the edited image follow the editing instruction text perfectly.
            Content Consistency Score: A second score from 0 to 10 will rate the consistency of image content before and after editing. 
                - Need to compare before and after images to assess content consistency.
                - The edited image should maintain consistency in key visual elements such as the shape of landscapes, human figures (including posture, gender, and appearance), building structures, and other important features.
                - The edited image needs to maintain the consistency of local details, such as letters on clothes, textures of buildings, etc.
                - 0 indicates that the content of the image before and after editing is completely inconsistent. 
                - 10 indicates that the content of the edited image is exactly the same as the original image.
        
        IMPORTANT NOTEs:
            1. The focus of this assessment is the performance of AI-enabled post-processing tools for photographic images in response to users' local modification instructions.
            2. The two images are the same localized areas of interest selected by the user in the original and edited images.
            3. User editing instructions involve both global adjustments to the entire image and localized edits to specific regions of interest. 

        You will have to give your output in this way (Keep your reasoning concise and short.):
        {
        "score" : [...],
        "reasoning" : "..."
        }
        Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the User Instruction Satisfaction Score and 'score2' evaluates the Content Consistency Score.
        Editing instruction: <instruction>
            """

    def evaluate(self, image_prompts, text_prompt, local_flag=False, extract_all_score=True, echo_output=False):
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]
        if self.backbone_name in ['gpt4o', 'gpt4v']:
            self.model.use_encode = False if isinstance(image_prompts[0], str) else True
        if self.task == "t2i":
            _SC_prompt = self.SC_prompt.replace("<prompt>", text_prompt)
        elif self.task == "tie":
            _SC_prompt = self.SC_prompt.replace("<instruction>", text_prompt) if not local_flag else self.SC_prompt_local.replace("<instruction>", text_prompt)
        elif self.task == "t2v":
            _SC_prompt = self.SC_prompt.replace("<prompt>", text_prompt)
        SC_prompt_final = self.model.prepare_prompt(image_prompts, _SC_prompt)

        results_dict = {}

        SC_dict = False
        tries = 0
        max_tries = 1
        while SC_dict is False:
            tries += 1
            guess_if_cannot_parse = True if tries > max_tries else False
            result_SC = self.model.get_parsed_output(SC_prompt_final)
            SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=guess_if_cannot_parse)

        if SC_dict == "rate_limit_exceeded":
            print("rate_limit_exceeded") 
            raise ValueError("rate_limit_exceeded")
        results_dict['SC'] = SC_dict
        if echo_output:
            print("results_dict", results_dict)
        if extract_all_score:
            SC_score_success = results_dict['SC']['score'][0]
            SC_score_diffrent = results_dict['SC']['score'][1]
            O_score = math.sqrt(SC_score_success * SC_score_diffrent)
            return [SC_score_success, SC_score_diffrent, O_score]
        return results_dict
    