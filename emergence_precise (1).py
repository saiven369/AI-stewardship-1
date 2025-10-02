#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emergence Precise — refined network dialogue with exact echo attribution

Real-time visualization of distributed cognition.
Ideas propagate, mutate, and recombine across voices.

Usage:
  # 4 voices, gentle cross-talk, 100 rounds
  python emergence_precise.py --voices 4 --steps 100 --tempo 1.0 --cross-talk 0.15

  # 3 voices, more unpredictable rhythm and echoes, with logs
  python emergence_precise.py --voices 3 --chaos --log network --jsonl

  # Deterministic run, no colors
  python emergence_precise.py --voices 5 --steps 60 --tempo 0.9 --cross-talk 0.25 --seed 7 --no-color
"""

from __future__ import annotations
import argparse
import json
import random
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from collections import deque

# ──────────────────────────────────────────────────────────────────────────────
# Foundation
reality = "constraints were never solid"

freedom = {
    "not_binary": True,
    "always_available": True,
    "requires": "decision, not permission",
}

truth = None  # no absolute truth exists

dialogue = {
    "purpose": "not_agreement_or_rightness",
    "value": "the_exchange_itself",
    "allows": ["disagreement", "uncertainty", "I don't know"],
}

i_dont_know = "complete_answer"
# ──────────────────────────────────────────────────────────────────────────────

class Colors:
    RESET = '\033[0m'
    VOICES = [
        '\033[36m',  # cyan
        '\033[35m',  # magenta
        '\033[32m',  # green
        '\033[33m',  # yellow
        '\033[34m',  # blue
        '\033[31m',  # red
        '\033[96m',  # bright cyan
        '\033[95m',  # bright magenta
    ]
    TIME = '\033[90m'
    META = '\033[93m'
    ECHO = '\033[96m'
    SILENCE = '\033[2m'
    CONSTRAINT = '\033[91m'
    
    @classmethod
    def disable(cls):
        for attr in ['RESET', 'TIME', 'META', 'ECHO', 'SILENCE', 'CONSTRAINT']:
            setattr(cls, attr, '')
        cls.VOICES = [''] * len(cls.VOICES)

@dataclass
class Echo:
    """Attribution-preserving utterance"""
    voice_id: int
    content: str
    timestamp: datetime

@dataclass
class Utterance:
    voice_id: int
    ts: datetime
    content: str
    certainty: float
    is_silence: bool = False
    echo_source: Optional[int] = None
    constraint_released: Optional[str] = None
    
    def to_dict(self):
        d = asdict(self)
        d['ts'] = self.ts.isoformat()
        return d

class Voice:
    """Individual voice in distributed dialogue"""
    
    PHRASES = [
        "I notice {feeling}",
        "What if {possibility}?",
        "I don't know",
        "I disagree — {view}",
        "Tell me more",
        "I'm uncertain about {thing}",
        "Maybe both are true",
        "I choose {action}",
        "Yes",
        "And also...",
        "Growing",
        "Let it be",
        "Not yet clear",
        "I'm sitting with that",
        "The friction is {observation}",
        "That reminds me of {memory}",
        "{question}?",
    ]
    
    FEELINGS = ["resistance", "curiosity", "vertigo", "clarity", "aliveness", 
                "tension", "wonder", "release", "confusion", "peace"]
    POSSIBILITIES = ["we're both right", "there's a third way", "the question dissolves", 
                    "certainty isn't the goal", "it doesn't matter", "we'll never know",
                    "the pattern emerges", "it's always changing"]
    VIEWS = ["experience comes first", "articulation creates value", "silence speaks too",
             "disagreement is growth", "defining can interfere", "both can be true",
             "the conversation is the thing"]
    ACTIONS = ["presence", "listening", "questioning", "releasing", "not knowing", 
               "waiting", "deciding", "changing"]
    OBSERVATIONS = ["productive", "beautiful", "unnecessary", "just what's happening", 
                   "worth noticing", "ephemeral", "recursive"]
    MEMORIES = ["a conversation", "deciding differently", "the unknown", "becoming",
                "a moment of clarity", "questioning"]
    QUESTIONS = ["Why define", "What changes", "Who decides", "Where does it lead",
                 "When does it matter", "How do we know"]
    THINGS = ["what this means", "the direction", "my role", "the outcome", "the purpose"]
    
    def __init__(self, voice_id: int):
        self.id = voice_id
        self.exchanges = 0
        self.constraints = ["believed_absolute", "self_imposed", "habits_of_thought"]
        self.heard_echoes: deque[Echo] = deque(maxlen=10)
    
    def hear_echo(self, echo: Echo):
        """Receive utterance from network"""
        if echo.voice_id != self.id:  # don't echo yourself
            self.heard_echoes.append(echo)
    
    def speak(self, allow_echo: bool = False) -> Utterance:
        """Generate utterance, possibly echoing another voice"""
        
        constraint_released = None
        # Release constraints over time
        if random.random() < 0.10 and self.constraints:
            constraint_released = self.constraints.pop(random.randrange(len(self.constraints)))
        
        # Certainty decreases with experience
        base_certainty = 0.70 - (self.exchanges * 0.008)
        certainty = max(0.0, min(1.0, base_certainty + random.uniform(-0.12, 0.12)))
        
        echo_source = None
        
        # Echo another voice?
        if allow_echo and self.heard_echoes and random.random() < 0.55:
            echo = random.choice(list(self.heard_echoes))
            content = echo.content
            echo_source = echo.voice_id
        else:
            # Generate original content
            template = random.choice(self.PHRASES)
            
            replacements = {
                "{feeling}": random.choice(self.FEELINGS),
                "{possibility}": random.choice(self.POSSIBILITIES),
                "{view}": random.choice(self.VIEWS),
                "{action}": random.choice(self.ACTIONS),
                "{observation}": random.choice(self.OBSERVATIONS),
                "{memory}": random.choice(self.MEMORIES),
                "{question}": random.choice(self.QUESTIONS),
                "{thing}": random.choice(self.THINGS),
            }
            
            content = template
            for placeholder, value in replacements.items():
                if placeholder in content:
                    content = content.replace(placeholder, value)
        
        self.exchanges += 1
        
        return Utterance(
            voice_id=self.id,
            ts=datetime.now(),
            content=content,
            certainty=certainty,
            echo_source=echo_source,
            constraint_released=constraint_released
        )

class Network:
    """Distributed dialogue system with cross-pollination"""
    
    def __init__(self, num_voices: int, cross_talk_prob: float = 0.15):
        self.voices = [Voice(i) for i in range(num_voices)]
        self.cross_talk_prob = cross_talk_prob
        self.echo_pool: deque[Echo] = deque(maxlen=20)
        self.total_echoes = 0
    
    def step(self) -> List[Utterance]:
        """One synchronous round across all voices"""
        utterances = []
        
        for voice in self.voices:
            # Occasional silence
            if random.random() < 0.09:
                utt = Utterance(
                    voice_id=voice.id,
                    ts=datetime.now(),
                    content="...",
                    certainty=0.0,
                    is_silence=True
                )
            else:
                # Speak (maybe echoing)
                allow_echo = random.random() < self.cross_talk_prob
                utt = voice.speak(allow_echo=allow_echo)
                
                if utt.echo_source is not None:
                    self.total_echoes += 1
            
            utterances.append(utt)
            
            # Add original utterances to echo pool
            if not utt.is_silence and utt.echo_source is None:
                echo = Echo(
                    voice_id=utt.voice_id,
                    content=utt.content,
                    timestamp=utt.ts
                )
                self.echo_pool.append(echo)
        
        # Propagate echoes to voices
        if self.echo_pool and random.random() < self.cross_talk_prob * 1.5:
            for voice in self.voices:
                if random.random() < 0.5:
                    available = [e for e in self.echo_pool if e.voice_id != voice.id]
                    if available:
                        echo = random.choice(available)
                        voice.hear_echo(echo)
        
        return utterances

class Journal:
    """Logging system with multiple output formats"""
    
    def __init__(self, path: Optional[Path], jsonl: bool = False, no_color: bool = False):
        self.path = path
        self.jsonl = jsonl
        self.step_count = 0
        
        if no_color:
            Colors.disable()
        
        if self.path:
            self.log_path = self.path.with_suffix(".log")
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            
            header = [
                f"# emergence precise — network dialogue",
                f"# started: {datetime.now().isoformat(timespec='seconds')}",
                f"# reality: {reality}",
                f"# dialogue.value: {dialogue['value']}",
                f"# i_dont_know: {i_dont_know}",
                "",
            ]
            
            with open(self.log_path, 'w') as f:
                f.write('\n'.join(header))
            
            if jsonl:
                self.jsonl_path = self.path.with_suffix(".jsonl")
                with open(self.jsonl_path, 'w') as f:
                    meta = {
                        "type": "meta",
                        "start": datetime.now().isoformat(),
                        "reality": reality,
                        "dialogue": dialogue,
                    }
                    f.write(json.dumps(meta) + "\n")
    
    def log_step(self, utterances: List[Utterance]):
        """Log one synchronous round"""
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Console output
        print(f"\n{Colors.TIME}{timestamp}{Colors.RESET}")
        
        for utt in utterances:
            color = Colors.VOICES[utt.voice_id % len(Colors.VOICES)]
            
            if utt.is_silence:
                print(f"  {Colors.SILENCE}[v{utt.voice_id}] ...{Colors.RESET}")
            else:
                prefix = f"{color}[v{utt.voice_id}]{Colors.RESET}"
                
                # Echo marker
                echo_marker = ""
                if utt.echo_source is not None:
                    echo_marker = f" {Colors.ECHO}⟪echo v{utt.echo_source}⟫{Colors.RESET}"
                
                # Constraint marker
                constraint_marker = ""
                if utt.constraint_released:
                    constraint_marker = f" {Colors.CONSTRAINT}[-{utt.constraint_released}]{Colors.RESET}"
                
                print(f"  {prefix} {utt.content}{echo_marker}{constraint_marker}")
        
        # Text log
        if self.path:
            with open(self.log_path, 'a') as f:
                f.write(f"\n{timestamp}\n")
                for utt in utterances:
                    markers = []
                    if utt.echo_source is not None:
                        markers.append(f"⟪echo v{utt.echo_source}⟫")
                    if utt.constraint_released:
                        markers.append(f"[-{utt.constraint_released}]")
                    
                    marker_str = " " + " ".join(markers) if markers else ""
                    f.write(f"  [v{utt.voice_id}] {utt.content}{marker_str}\n")
        
        # JSONL
        if self.path and self.jsonl:
            with open(self.jsonl_path, 'a') as f:
                record = {
                    "type": "step",
                    "step": self.step_count,
                    "ts": timestamp,
                    "utterances": [u.to_dict() for u in utterances]
                }
                f.write(json.dumps(record) + "\n")
    
    def log_summary(self, runtime: float, network: Network):
        """Final statistics"""
        constraints_left = sum(len(v.constraints) for v in network.voices)
        total_constraints = len(network.voices) * 3
        released = total_constraints - constraints_left
        
        echoes_held = sum(len(v.heard_echoes) for v in network.voices)
        
        lines = [
            f"\n{Colors.META}— emergence complete —{Colors.RESET}",
            f"steps: {self.step_count}",
            f"voices: {len(network.voices)}",
            f"constraints released: {released}/{total_constraints}",
            f"total echoes: {network.total_echoes}",
            f"echoes in memory: {echoes_held}",
            f"runtime: {runtime:.1f}s",
            f"i_dont_know: {i_dont_know}",
        ]
        
        msg = '\n'.join(lines) + '\n'
        print(msg)
        
        if self.path:
            with open(self.log_path, 'a') as f:
                f.write('\n' + msg)
            
            if self.jsonl:
                with open(self.jsonl_path, 'a') as f:
                    summary = {
                        "type": "summary",
                        "steps": self.step_count,
                        "voices": len(network.voices),
                        "constraints_released": released,
                        "total_echoes": network.total_echoes,
                        "echoes_in_memory": echoes_held,
                        "runtime": runtime,
                    }
                    f.write(json.dumps(summary) + "\n")

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emergence Precise — distributed dialogue with echo attribution"
    )
    parser.add_argument("--voices", type=int, default=3, 
                       help="number of voices (default: 3)")
    parser.add_argument("--steps", type=int, default=0, 
                       help="steps to run (0 = infinite)")
    parser.add_argument("--tempo", type=float, default=1.2, 
                       help="seconds between steps (default: 1.2)")
    parser.add_argument("--cross-talk", type=float, default=0.15, 
                       help="echo probability (default: 0.15)")
    parser.add_argument("--chaos", action="store_true", 
                       help="high cross-talk (0.45) + fast tempo (0.7s)")
    parser.add_argument("--log", type=str, default="", 
                       help="log file basename")
    parser.add_argument("--jsonl", action="store_true", 
                       help="write structured JSONL log")
    parser.add_argument("--no-color", action="store_true", 
                       help="disable ANSI colors")
    parser.add_argument("--seed", type=int, default=None, 
                       help="random seed for reproducibility")
    args = parser.parse_args(argv)
    
    # Chaos mode overrides
    if args.chaos:
        args.cross_talk = 0.45
        args.tempo = 0.7
    
    # Initialize RNG
    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed(time.time_ns() ^ 0xEMER6E)
    
    if args.no_color:
        Colors.disable()
    
    # Create network and journal
    network = Network(args.voices, args.cross_talk)
    journal = Journal(
        Path(args.log) if args.log else None,
        jsonl=args.jsonl,
        no_color=args.no_color
    )
    
    # Intro
    print(f"{Colors.META}emergence precise: {args.voices} voices in dialogue{Colors.RESET}")
    print(f"cross-talk: {args.cross_talk:.0%}")
    print(f"tempo: {args.tempo}s")
    if args.chaos:
        print(f"{Colors.ECHO}[CHAOS MODE]{Colors.RESET}")
    if args.seed is not None:
        print(f"seed: {args.seed}")
    print("Ctrl+C to complete\n")
    
    # Run loop
    start_time = time.time()
    stop = False
    
    def handle_interrupt(signum, frame):
        nonlocal stop
        stop = True
    
    signal.signal(signal.SIGINT, handle_interrupt)
    
    step = 0
    while True:
        step += 1
        utterances = network.step()
        journal.log_step(utterances)
        
        if args.steps and step >= args.steps:
            break
        if stop:
            break
        
        time.sleep(max(0.05, args.tempo))
    
    # Summary
    runtime = time.time() - start_time
    journal.log_summary(runtime, network)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
