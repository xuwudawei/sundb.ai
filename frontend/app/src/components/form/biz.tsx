import { type EmbeddingModel } from '@/api/embedding-models';
import { type KnowledgeBaseSummary } from '@/api/knowledge-base';
import { type LLM } from '@/api/llms';
import type { ProviderOption } from '@/api/providers';
import { type Reranker } from '@/api/rerankers';
import { CreateEmbeddingModelForm } from '@/components/embedding-models/CreateEmbeddingModelForm';
import { useAllEmbeddingModels } from '@/components/embedding-models/hooks';
import { FormCombobox, type FormComboboxConfig, type FormComboboxProps } from '@/components/form/control-widget';
import { useAllKnowledgeBases } from '@/components/knowledge-base/hooks';
import { CreateLLMForm } from '@/components/llm/CreateLLMForm';
import { useAllLlms } from '@/components/llm/hooks';
import { ManagedDialog } from '@/components/managed-dialog';
import { ManagedPanelContext } from '@/components/managed-panel';
import { CreateRerankerForm } from '@/components/reranker/CreateRerankerForm';
import { useAllRerankers } from '@/components/reranker/hooks';
import { DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { AlertTriangleIcon, DotIcon, PlusIcon } from 'lucide-react';
import { forwardRef } from 'react';

export const EmbeddingModelSelect = forwardRef<any, Omit<FormComboboxProps, 'config'> & { reverse?: boolean }>(({ reverse = true, ...props }, ref) => {
  const { data: embeddingModels, isLoading, mutate, error } = useAllEmbeddingModels();

  return (
    <FormCombobox
      {...props}
      placeholder="Default Embedding Model"
      config={{
        options: embeddingModels ?? [],
        optionKeywords: option => [option.name, option.provider, option.model],
        loading: isLoading,
        error,
        renderValue: option => (<span>{option.name} <span className="text-muted-foreground">[{option.vector_dimension}]</span></span>),
        renderOption: option => (
          <div>
            <div><strong>{option.name}</strong></div>
            <div className="text-xs text-muted-foreground">
              <strong>{option.provider}</strong>:{option.model} [{option.vector_dimension}]
            </div>
          </div>
        ),
        renderCreateOption: (Wrapper, onCreated) => (
          <ManagedDialog>
            <ManagedPanelContext.Consumer>
              {({ setOpen }) => (
                <>
                  <Wrapper onSelect={() => setOpen(true)}>
                    <span className="flex gap-1 items-center text-muted-foreground">
                      <PlusIcon className="size-4" />
                      Create New Embedding Model
                    </span>
                  </Wrapper>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>
                        Create New Embedding Model
                      </DialogTitle>
                      <DialogDescription />
                    </DialogHeader>
                    <CreateEmbeddingModelForm
                      onCreated={embeddingModel => {
                        mutate();
                        onCreated(embeddingModel);
                        setOpen(false);
                      }}
                    />
                  </DialogContent>
                </>
              )}
            </ManagedPanelContext.Consumer>
          </ManagedDialog>
        ),
        key: 'id',
      } satisfies FormComboboxConfig<EmbeddingModel>}
    />
  );
});

EmbeddingModelSelect.displayName = 'EmbeddingModelSelect';

export const LLMSelect = forwardRef<any, Omit<FormComboboxProps, 'config'> & { reverse?: boolean }>(({ reverse = true, ...props }, ref) => {
  const { data: llms, isLoading, mutate, error } = useAllLlms();

  return (
    <FormCombobox
      {...props}
      placeholder="Default LLM"
      config={{
        options: llms ?? [],
        loading: isLoading,
        error,
        renderValue: option => (<span>{option.name}</span>),
        renderOption: option => (
          <div>
            <div><strong>{option.name}</strong></div>
            <div className="text-xs text-muted-foreground">
              <strong>{option.provider}</strong>:{option.model}
            </div>
          </div>
        ),
        renderCreateOption: (Wrapper, onCreated) => (
          <ManagedDialog>
            <ManagedPanelContext.Consumer>
              {({ setOpen }) => (
                <>
                  <Wrapper onSelect={() => setOpen(true)}>
                    <span className="flex gap-1 items-center text-muted-foreground">
                      <PlusIcon className="size-4" />
                      Create New LLM
                    </span>
                  </Wrapper>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>
                        Create New LLM
                      </DialogTitle>
                      <DialogDescription />
                    </DialogHeader>
                    <CreateLLMForm
                      onCreated={llm => {
                        mutate();
                        onCreated(llm);
                        setOpen(false);
                      }}
                    />
                  </DialogContent>
                </>
              )}
            </ManagedPanelContext.Consumer>
          </ManagedDialog>
        ),
        optionKeywords: option => [option.name, option.provider, option.model],
        key: 'id',
      } satisfies FormComboboxConfig<LLM>}
    />
  );
});

LLMSelect.displayName = 'LLMSelect';

export const RerankerSelect = forwardRef<any, Omit<FormComboboxProps, 'config'> & { reverse?: boolean }>(({ reverse = true, ...props }, ref) => {
  const { data: rerankers, mutate, isLoading, error } = useAllRerankers();

  return (
    <FormCombobox
      {...props}
      placeholder="Default Reranker Model"
      config={{
        options: rerankers ?? [],
        optionKeywords: option => [option.name, option.provider, option.model],
        loading: isLoading,
        error,
        renderValue: option => (<span>{option.name}</span>),
        renderOption: option => (
          <div>
            <div><strong>{option.name}</strong></div>
            <div className="text-xs text-muted-foreground">
              <strong>{option.provider}</strong>:{option.model}
            </div>
          </div>
        ),
        renderCreateOption: (Wrapper, onCreated) => (
          <ManagedDialog>
            <ManagedPanelContext.Consumer>
              {({ setOpen }) => (
                <>
                  <Wrapper onSelect={() => setOpen(true)}>
                    <span className="flex gap-1 items-center text-muted-foreground">
                      <PlusIcon className="size-4" />
                      Create New Reranker
                    </span>
                  </Wrapper>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>
                        Create New Reranker
                      </DialogTitle>
                      <DialogDescription />
                    </DialogHeader>
                    <CreateRerankerForm
                      onCreated={reranker => {
                        mutate();
                        onCreated(reranker);
                        setOpen(false);
                      }}
                    />
                  </DialogContent>
                </>
              )}
            </ManagedPanelContext.Consumer>
          </ManagedDialog>
        ),
        key: 'id',
      } satisfies FormComboboxConfig<Reranker>}
    />
  );
});

RerankerSelect.displayName = 'RerankerSelect';

export interface ProviderSelectProps<Provider extends ProviderOption = ProviderOption> extends Omit<FormComboboxProps, 'config'> {
  options: ProviderOption[] | undefined;
  isLoading: boolean;
  error: unknown;
}

export const ProviderSelect = forwardRef<any, ProviderSelectProps>(({
  options, isLoading, error, ...props
}, ref) => {
  return (
    <FormCombobox
      ref={ref}
      config={{
        options: options ?? [],
        optionKeywords: option => [option.provider, option.provider_description ?? '', option.provider_display_name ?? ''],
        loading: isLoading,
        error,
        renderOption: option => (
          <div>
            <div className="text-sm font-bold">{option.provider_display_name ?? option.provider}</div>
            {option.provider_description && <div className="text-xs text-muted-foreground break-words" style={{ maxWidth: 'calc(var(--radix-select-trigger-width) - 68px)' }}>{option.provider_description}</div>}
          </div>
        ),
        itemClassName: 'space-y-1',
        renderValue: option => option.provider_display_name ?? option.provider,
        key: 'provider',
      } satisfies FormComboboxConfig<ProviderOption>}
      contentWidth="anchor"
      {...props}
    />
  );
});

ProviderSelect.displayName = 'ProviderSelect';

export const KBSelect = forwardRef<any, Omit<FormComboboxProps, 'config'> & { reverse?: boolean }>(({ reverse = true, ...props }, ref) => {
  const { data: kbs, isLoading, error } = useAllKnowledgeBases();

  return (
    <FormCombobox
      ref={ref}
      {...props}
      placeholder="Select Knowledge Base"
      config={{
        options: kbs ?? [],
        optionKeywords: option => [String(option.id), option.name, option.description],
        loading: isLoading,
        error,
        renderValue: option => (
          <div className="">
            <span>{option.name}</span>
            <div className="text-xs text-muted-foreground ml-2 inline-flex gap-1 items-center">
              <span>
                {(option.documents_total ?? 0) || <><AlertTriangleIcon className="text-yellow-600 dark:text-yellow-400 inline-flex size-3 mr-0.5" /> no</>} documents
              </span>
              <DotIcon className="size-4" />
              <span className="text-xs text-muted-foreground">
                {(option.data_sources_total ?? 0) || <><AlertTriangleIcon className="inline-flex size-3 mr-0.5" /> no</>} data sources
              </span>
            </div>
          </div>
        ),
        renderOption: option => (
          <div className="space-y-1">
            <div>
              <strong>
                {option.name}
              </strong>
            </div>
            <div className="text-xs text-muted-foreground flex gap-1 items-center">
              <span>
                {(option.documents_total ?? 0) || <><AlertTriangleIcon className="text-yellow-600 dark:text-yellow-400 inline-flex size-3 mr-0.5" /> no</>} documents
              </span>
              <DotIcon className="size-4" />
              <span>
                {(option.data_sources_total ?? 0) || <><AlertTriangleIcon className="inline-flex size-3 mr-0.5" /> no</>} data sources
              </span>
            </div>
            <div className="text-xs text-muted-foreground">
              {option.description}
            </div>
          </div>
        ),
        key: 'id',
      } satisfies FormComboboxConfig<KnowledgeBaseSummary>}
    />
  );
});

KBSelect.displayName = 'KBSelect';
